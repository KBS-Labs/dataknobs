"""Tests for tools/kb_tools.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from dataknobs_llm.tools.context import ToolExecutionContext, WizardStateSnapshot

from dataknobs_bots.tools.kb_tools import (
    AddKBResourceTool,
    CheckKnowledgeSourceTool,
    IngestKnowledgeBaseTool,
    ListKBResourcesTool,
    RemoveKBResourceTool,
)


def _make_context(
    wizard_data: dict[str, Any] | None = None,
) -> ToolExecutionContext:
    """Create a ToolExecutionContext with wizard state."""
    if wizard_data is not None:
        wizard_state = WizardStateSnapshot(
            current_stage="test",
            collected_data=wizard_data,
            history=["test"],
            completed=False,
        )
    else:
        wizard_state = None
    return ToolExecutionContext(
        conversation_id="test-conv",
        user_id="test-user",
        wizard_state=wizard_state,
    )


class TestCheckKnowledgeSourceTool:
    """Tests for CheckKnowledgeSourceTool."""

    @pytest.mark.asyncio
    async def test_valid_directory(self, tmp_path: Path) -> None:
        # Create some knowledge files
        (tmp_path / "intro.md").write_text("# Intro")
        (tmp_path / "guide.txt").write_text("Guide content")
        (tmp_path / "data.csv").write_text("a,b\n1,2")
        (tmp_path / "ignore.py").write_text("# not a doc")

        wizard_data: dict[str, Any] = {}
        tool = CheckKnowledgeSourceTool()
        result = await tool.execute_with_context(
            _make_context(wizard_data), source_path=str(tmp_path)
        )

        assert result["exists"] is True
        assert result["file_count"] == 3
        assert "intro.md" in result["files_found"]
        assert "guide.txt" in result["files_found"]
        assert "data.csv" in result["files_found"]
        assert "ignore.py" not in result["files_found"]

        # Verify wizard data was updated
        assert wizard_data["source_verified"] is True
        assert wizard_data["_source_path_resolved"] == str(tmp_path)

    @pytest.mark.asyncio
    async def test_auto_populates_kb_resources(self, tmp_path: Path) -> None:
        """Check auto-populates _kb_resources with discovered files."""
        (tmp_path / "intro.md").write_text("# Intro")
        (tmp_path / "guide.txt").write_text("Guide content")

        wizard_data: dict[str, Any] = {}
        tool = CheckKnowledgeSourceTool()
        await tool.execute_with_context(
            _make_context(wizard_data), source_path=str(tmp_path)
        )

        resources = wizard_data["_kb_resources"]
        assert len(resources) == 2
        paths = {r["path"] for r in resources}
        assert "intro.md" in paths
        assert "guide.txt" in paths
        # Each entry has source and type
        for r in resources:
            assert r["type"] == "file"
            assert "source" in r

    @pytest.mark.asyncio
    async def test_missing_directory(self) -> None:
        wizard_data: dict[str, Any] = {}
        tool = CheckKnowledgeSourceTool()
        result = await tool.execute_with_context(
            _make_context(wizard_data),
            source_path="/nonexistent/path/abc123",
        )

        assert result["exists"] is False
        assert "error" in result
        assert wizard_data["source_verified"] is False

    @pytest.mark.asyncio
    async def test_no_matching_files(self, tmp_path: Path) -> None:
        # Directory exists but has no matching files
        (tmp_path / "code.py").write_text("x = 1")

        wizard_data: dict[str, Any] = {}
        tool = CheckKnowledgeSourceTool()
        result = await tool.execute_with_context(
            _make_context(wizard_data), source_path=str(tmp_path)
        )

        assert result["exists"] is True
        assert result["file_count"] == 0
        assert wizard_data["source_verified"] is True

    @pytest.mark.asyncio
    async def test_custom_glob_patterns(self, tmp_path: Path) -> None:
        (tmp_path / "data.py").write_text("x = 1")
        (tmp_path / "test.py").write_text("y = 2")
        (tmp_path / "notes.md").write_text("# Notes")

        wizard_data: dict[str, Any] = {}
        tool = CheckKnowledgeSourceTool()
        result = await tool.execute_with_context(
            _make_context(wizard_data),
            source_path=str(tmp_path),
            file_patterns=["*.py"],
        )

        assert result["file_count"] == 2
        assert "data.py" in result["files_found"]
        assert "notes.md" not in result["files_found"]

    @pytest.mark.asyncio
    async def test_preserves_existing_kb_resources(self, tmp_path: Path) -> None:
        """If _kb_resources already has entries, appends new ones."""
        (tmp_path / "doc.md").write_text("content")
        existing = [{"path": "prev.md", "type": "file"}]
        wizard_data: dict[str, Any] = {"_kb_resources": existing}

        tool = CheckKnowledgeSourceTool()
        await tool.execute_with_context(
            _make_context(wizard_data), source_path=str(tmp_path)
        )

        # Should have original + newly discovered
        assert len(wizard_data["_kb_resources"]) == 2
        paths = {r["path"] for r in wizard_data["_kb_resources"]}
        assert "prev.md" in paths
        assert "doc.md" in paths

    @pytest.mark.asyncio
    async def test_schema(self) -> None:
        tool = CheckKnowledgeSourceTool()
        assert "source_path" in tool.schema["properties"]
        assert "source_path" in tool.schema["required"]


class TestListKBResourcesTool:
    """Tests for ListKBResourcesTool."""

    @pytest.mark.asyncio
    async def test_empty_resources(self) -> None:
        wizard_data: dict[str, Any] = {}
        tool = ListKBResourcesTool()
        result = await tool.execute_with_context(_make_context(wizard_data))

        assert result["count"] == 0
        assert result["resources"] == []
        assert result["source_path"] is None

    @pytest.mark.asyncio
    async def test_with_resources(self) -> None:
        wizard_data: dict[str, Any] = {
            "_kb_resources": [
                {"path": "intro.md", "type": "file"},
                {"path": "notes.txt", "type": "inline"},
            ],
            "_source_path_resolved": "/data/knowledge",
        }
        tool = ListKBResourcesTool()
        result = await tool.execute_with_context(_make_context(wizard_data))

        assert result["count"] == 2
        assert result["source_path"] == "/data/knowledge"
        paths = [r["path"] for r in result["resources"]]
        assert "intro.md" in paths
        assert "notes.txt" in paths


class TestAddKBResourceTool:
    """Tests for AddKBResourceTool."""

    @pytest.mark.asyncio
    async def test_add_file_resource(self) -> None:
        wizard_data: dict[str, Any] = {"_kb_resources": []}
        tool = AddKBResourceTool()
        result = await tool.execute_with_context(
            _make_context(wizard_data),
            path="guide.md",
            resource_type="file",
            description="User guide",
        )

        assert result["success"] is True
        assert result["total_resources"] == 1
        assert wizard_data["_kb_resources"][0]["path"] == "guide.md"
        assert wizard_data["_kb_resources"][0]["description"] == "User guide"

    @pytest.mark.asyncio
    async def test_add_with_title(self) -> None:
        wizard_data: dict[str, Any] = {"_kb_resources": []}
        tool = AddKBResourceTool()
        result = await tool.execute_with_context(
            _make_context(wizard_data),
            path="guide.md",
            title="User Guide",
        )

        assert result["success"] is True
        assert wizard_data["_kb_resources"][0]["title"] == "User Guide"

    @pytest.mark.asyncio
    async def test_add_duplicate_rejected(self) -> None:
        wizard_data: dict[str, Any] = {
            "_kb_resources": [{"path": "guide.md", "type": "file"}]
        }
        tool = AddKBResourceTool()
        result = await tool.execute_with_context(
            _make_context(wizard_data), path="guide.md"
        )

        assert result["success"] is False
        assert "already exists" in result["error"]

    @pytest.mark.asyncio
    async def test_add_inline_content(self, tmp_path: Path) -> None:
        wizard_data: dict[str, Any] = {
            "_kb_resources": [],
            "domain_id": "test-bot",
        }
        tool = AddKBResourceTool(knowledge_dir=tmp_path)
        result = await tool.execute_with_context(
            _make_context(wizard_data),
            path="faq.md",
            resource_type="inline",
            content="# FAQ\n\nQ: How? A: Like this.",
        )

        assert result["success"] is True
        # Verify file was written
        written = tmp_path / "test-bot" / "faq.md"
        assert written.exists()
        assert "FAQ" in written.read_text()
        assert wizard_data["_kb_resources"][0]["source"] == str(written)

    @pytest.mark.asyncio
    async def test_inline_no_content_error(self) -> None:
        wizard_data: dict[str, Any] = {"_kb_resources": []}
        tool = AddKBResourceTool()
        result = await tool.execute_with_context(
            _make_context(wizard_data),
            path="empty.md",
            resource_type="inline",
        )

        assert result["success"] is False
        assert "Content is required" in result["error"]

    @pytest.mark.asyncio
    async def test_inline_no_knowledge_dir_error(self) -> None:
        wizard_data: dict[str, Any] = {"_kb_resources": []}
        tool = AddKBResourceTool()  # no knowledge_dir
        result = await tool.execute_with_context(
            _make_context(wizard_data),
            path="data.md",
            resource_type="inline",
            content="some content",
        )

        assert result["success"] is False
        assert "knowledge directory" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_initializes_kb_resources_if_absent(self) -> None:
        wizard_data: dict[str, Any] = {}
        tool = AddKBResourceTool()
        result = await tool.execute_with_context(
            _make_context(wizard_data), path="doc.md"
        )

        assert result["success"] is True
        assert len(wizard_data["_kb_resources"]) == 1

    @pytest.mark.asyncio
    async def test_knowledge_dir_from_wizard_data(self, tmp_path: Path) -> None:
        """Knowledge dir resolved from wizard data _knowledge_dir."""
        wizard_data: dict[str, Any] = {
            "_kb_resources": [],
            "_knowledge_dir": str(tmp_path),
            "domain_id": "wd-bot",
        }
        tool = AddKBResourceTool()  # no constructor dir
        result = await tool.execute_with_context(
            _make_context(wizard_data),
            path="content.md",
            resource_type="inline",
            content="Hello",
        )

        assert result["success"] is True
        assert (tmp_path / "wd-bot" / "content.md").exists()


class TestRemoveKBResourceTool:
    """Tests for RemoveKBResourceTool."""

    @pytest.mark.asyncio
    async def test_remove_existing(self) -> None:
        wizard_data: dict[str, Any] = {
            "_kb_resources": [
                {"path": "a.md", "type": "file"},
                {"path": "b.md", "type": "file"},
            ]
        }
        tool = RemoveKBResourceTool()
        result = await tool.execute_with_context(
            _make_context(wizard_data), path="a.md"
        )

        assert result["success"] is True
        assert result["remaining_resources"] == 1
        assert len(wizard_data["_kb_resources"]) == 1
        assert wizard_data["_kb_resources"][0]["path"] == "b.md"

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self) -> None:
        wizard_data: dict[str, Any] = {
            "_kb_resources": [{"path": "a.md", "type": "file"}]
        }
        tool = RemoveKBResourceTool()
        result = await tool.execute_with_context(
            _make_context(wizard_data), path="missing.md"
        )

        assert result["success"] is False
        assert "not found" in result["error"]
        assert "a.md" in result["available"]

    @pytest.mark.asyncio
    async def test_remove_from_empty(self) -> None:
        wizard_data: dict[str, Any] = {}
        tool = RemoveKBResourceTool()
        result = await tool.execute_with_context(
            _make_context(wizard_data), path="any.md"
        )

        assert result["success"] is False


class TestIngestKnowledgeBaseTool:
    """Tests for IngestKnowledgeBaseTool."""

    @pytest.mark.asyncio
    async def test_write_manifest(self, tmp_path: Path) -> None:
        wizard_data: dict[str, Any] = {
            "domain_id": "test-domain",
            "_kb_resources": [
                {"path": "intro.md", "type": "file"},
                {"path": "guide.txt", "type": "file"},
            ],
            "_source_path_resolved": "/data/source",
        }
        tool = IngestKnowledgeBaseTool(knowledge_dir=tmp_path)
        result = await tool.execute_with_context(
            _make_context(wizard_data),
            chunk_size=256,
            chunk_overlap=32,
        )

        assert result["success"] is True
        assert result["resource_count"] == 2
        assert result["chunk_size"] == 256

        # Verify manifest was written
        manifest_path = tmp_path / "test-domain" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["domain_id"] == "test-domain"
        assert len(manifest["resources"]) == 2
        assert manifest["chunking"]["chunk_size"] == 256
        assert manifest["chunking"]["chunk_overlap"] == 32

        # Verify wizard data updated
        assert wizard_data["ingestion_complete"] is True
        assert wizard_data["kb_config"]["enabled"] is True
        assert wizard_data["kb_config"]["documents_path"] == str(
            tmp_path / "test-domain"
        )
        assert wizard_data["kb_resources"] == wizard_data["_kb_resources"]

    @pytest.mark.asyncio
    async def test_fallback_to_files_found(self, tmp_path: Path) -> None:
        """When no _kb_resources, falls back to files_found."""
        wizard_data: dict[str, Any] = {
            "domain_id": "fallback-domain",
            "files_found": ["readme.md", "notes.txt"],
        }
        tool = IngestKnowledgeBaseTool(knowledge_dir=tmp_path)
        result = await tool.execute_with_context(_make_context(wizard_data))

        assert result["success"] is True
        assert result["resource_count"] == 2

    @pytest.mark.asyncio
    async def test_no_resources_error(self, tmp_path: Path) -> None:
        wizard_data: dict[str, Any] = {"domain_id": "empty"}
        tool = IngestKnowledgeBaseTool(knowledge_dir=tmp_path)
        result = await tool.execute_with_context(_make_context(wizard_data))

        assert result["success"] is False
        assert "No resources" in result["error"]

    @pytest.mark.asyncio
    async def test_no_knowledge_dir_error(self) -> None:
        wizard_data: dict[str, Any] = {
            "domain_id": "test",
            "_kb_resources": [{"path": "a.md", "type": "file"}],
        }
        tool = IngestKnowledgeBaseTool()  # no knowledge_dir
        result = await tool.execute_with_context(_make_context(wizard_data))

        assert result["success"] is False
        assert "knowledge directory" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_default_chunking_params(self, tmp_path: Path) -> None:
        wizard_data: dict[str, Any] = {
            "domain_id": "defaults",
            "_kb_resources": [{"path": "doc.md", "type": "file"}],
        }
        tool = IngestKnowledgeBaseTool(knowledge_dir=tmp_path)
        result = await tool.execute_with_context(_make_context(wizard_data))

        assert result["success"] is True
        assert result["chunk_size"] == 512
        assert result["chunk_overlap"] == 50

    @pytest.mark.asyncio
    async def test_knowledge_dir_from_wizard_data(self, tmp_path: Path) -> None:
        """Knowledge dir resolved from wizard data _knowledge_dir."""
        wizard_data: dict[str, Any] = {
            "domain_id": "wd-domain",
            "_knowledge_dir": str(tmp_path),
            "_kb_resources": [{"path": "doc.md", "type": "file"}],
        }
        tool = IngestKnowledgeBaseTool()  # no constructor dir
        result = await tool.execute_with_context(_make_context(wizard_data))

        assert result["success"] is True
        assert (tmp_path / "wd-domain" / "manifest.json").exists()

    @pytest.mark.asyncio
    async def test_schema(self) -> None:
        tool = IngestKnowledgeBaseTool()
        assert "chunk_size" in tool.schema["properties"]
        assert "chunk_overlap" in tool.schema["properties"]
