"""Tests for tools/catalog_tools.py."""

from __future__ import annotations

from typing import Any

import pytest
from dataknobs_data.backends.memory import SyncMemoryDatabase
from dataknobs_llm.tools.context import ToolExecutionContext

from dataknobs_bots.memory.artifact_bank import ArtifactBank
from dataknobs_bots.memory.bank import MemoryBank
from dataknobs_bots.memory.catalog import ArtifactBankCatalog
from dataknobs_bots.tools.catalog_tools import (
    ListCatalogTool,
    LoadFromCatalogTool,
    SaveToCatalogTool,
)


def _make_artifact(
    name: str = "recipe",
    recipe_name: str | None = None,
    ingredients: list[dict[str, Any]] | None = None,
) -> ArtifactBank:
    """Create a minimal ArtifactBank."""
    section_db = SyncMemoryDatabase()
    sections = {
        "ingredients": MemoryBank(
            name="ingredients",
            schema={"required": ["name"]},
            db=section_db,
        ),
    }
    artifact = ArtifactBank(
        name=name,
        field_defs={"recipe_name": {"required": True}},
        sections=sections,
    )
    if recipe_name:
        artifact.set_field("recipe_name", recipe_name)
    if ingredients:
        bank = artifact.sections["ingredients"]
        for ing in ingredients:
            bank.add(ing, source_stage="test")
    return artifact


def _make_context(
    artifact: ArtifactBank | None = None,
    catalog: ArtifactBankCatalog | None = None,
) -> ToolExecutionContext:
    """Create a ToolExecutionContext with optional artifact and catalog."""
    extra: dict[str, Any] = {}
    if artifact is not None:
        extra["artifact"] = artifact
    if catalog is not None:
        extra["catalog"] = catalog
    return ToolExecutionContext(
        conversation_id="test-conv",
        user_id="test-user",
        extra=extra,
    )


class TestListCatalogTool:
    """Tests for ListCatalogTool."""

    @pytest.mark.asyncio
    async def test_empty_catalog(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        context = _make_context(catalog=catalog)
        tool = ListCatalogTool()

        result = await tool.execute_with_context(context)
        assert result["count"] == 0
        assert result["entries"] == []

    @pytest.mark.asyncio
    async def test_populated_catalog(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        artifact = _make_artifact(
            recipe_name="Cookies",
            ingredients=[{"name": "flour"}],
        )
        catalog.save(artifact)

        context = _make_context(catalog=catalog)
        tool = ListCatalogTool()

        result = await tool.execute_with_context(context)
        assert result["count"] == 1
        assert result["entries"][0]["name"] == "recipe"

    @pytest.mark.asyncio
    async def test_no_catalog_in_context(self) -> None:
        context = _make_context()
        tool = ListCatalogTool()

        result = await tool.execute_with_context(context)
        assert result["success"] is False
        assert "catalog" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_custom_tool_name(self) -> None:
        tool = ListCatalogTool(tool_name="my_list")
        assert tool.name == "my_list"


class TestSaveToCatalogTool:
    """Tests for SaveToCatalogTool."""

    @pytest.mark.asyncio
    async def test_save_valid(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        artifact = _make_artifact(
            recipe_name="Cookies",
            ingredients=[{"name": "flour"}],
        )
        context = _make_context(artifact=artifact, catalog=catalog)
        tool = SaveToCatalogTool()

        result = await tool.execute_with_context(context)
        assert result["success"] is True
        assert result["name"] == "recipe"
        assert result["catalog_count"] == 1

    @pytest.mark.asyncio
    async def test_save_validation_failure(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        artifact = _make_artifact()  # No recipe_name, no ingredients
        context = _make_context(artifact=artifact, catalog=catalog)
        tool = SaveToCatalogTool()

        result = await tool.execute_with_context(context)
        assert result["success"] is False
        assert "errors" in result

    @pytest.mark.asyncio
    async def test_save_no_artifact(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        context = _make_context(catalog=catalog)
        tool = SaveToCatalogTool()

        result = await tool.execute_with_context(context)
        assert result["success"] is False
        assert "artifact" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_save_no_catalog(self) -> None:
        artifact = _make_artifact(
            recipe_name="Cookies",
            ingredients=[{"name": "flour"}],
        )
        context = _make_context(artifact=artifact)
        tool = SaveToCatalogTool()

        result = await tool.execute_with_context(context)
        assert result["success"] is False
        assert "catalog" in result["error"].lower()


class TestLoadFromCatalogTool:
    """Tests for LoadFromCatalogTool."""

    @pytest.mark.asyncio
    async def test_load_existing(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        source = _make_artifact(
            recipe_name="Cookies",
            ingredients=[{"name": "flour"}, {"name": "sugar"}],
        )
        catalog.save(source)

        # Fresh target
        target = _make_artifact()
        context = _make_context(artifact=target, catalog=catalog)
        tool = LoadFromCatalogTool()

        result = await tool.execute_with_context(context, name="recipe")
        assert result["success"] is True
        assert result["loaded"]["name"] == "recipe"
        assert result["loaded"]["fields"]["recipe_name"] == "Cookies"
        assert result["loaded"]["sections"]["ingredients"] == 2

        # Verify target was actually populated
        assert target.field("recipe_name") == "Cookies"
        assert target.sections["ingredients"].count() == 2

    @pytest.mark.asyncio
    async def test_load_not_found(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        target = _make_artifact()
        context = _make_context(artifact=target, catalog=catalog)
        tool = LoadFromCatalogTool()

        result = await tool.execute_with_context(context, name="nonexistent")
        assert result["success"] is False
        assert "nonexistent" in result["error"]

    @pytest.mark.asyncio
    async def test_load_missing_name(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        target = _make_artifact()
        context = _make_context(artifact=target, catalog=catalog)
        tool = LoadFromCatalogTool()

        result = await tool.execute_with_context(context)
        assert result["success"] is False
        assert "name" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_load_no_catalog(self) -> None:
        target = _make_artifact()
        context = _make_context(artifact=target)
        tool = LoadFromCatalogTool()

        result = await tool.execute_with_context(context, name="recipe")
        assert result["success"] is False
        assert "catalog" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_load_no_artifact(self) -> None:
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        context = _make_context(catalog=catalog)
        tool = LoadFromCatalogTool()

        result = await tool.execute_with_context(context, name="recipe")
        assert result["success"] is False
        assert "artifact" in result["error"].lower()
