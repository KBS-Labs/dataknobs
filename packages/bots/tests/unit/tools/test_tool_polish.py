"""Tests for 03c Phase 1: Tool Polish.

Verifies:
- Response format consistency (all tools use success_response())
- Effects metadata in all catalog_metadata() methods
- Updated tool descriptions per 02b §5.2
"""

from __future__ import annotations

from typing import Any

import pytest
from dataknobs_data.backends.memory import SyncMemoryDatabase
from dataknobs_llm.tools.context import ToolExecutionContext

from dataknobs_bots.memory.artifact_bank import ArtifactBank
from dataknobs_bots.memory.bank import MemoryBank
from dataknobs_bots.memory.catalog import ArtifactBankCatalog
from dataknobs_bots.tools.bank_tools import (
    AddBankRecordTool,
    CompileArtifactTool,
    CompleteWizardTool,
    FinalizeBankTool,
    FinalizeArtifactTool,
    ListBankRecordsTool,
    RemoveBankRecordTool,
    RestartWizardTool,
    UpdateBankRecordTool,
)
from dataknobs_bots.tools.catalog_tools import (
    ListCatalogTool,
    LoadFromCatalogTool,
    SaveToCatalogTool,
)

# Valid effect categories per 02b P5a
VALID_EFFECTS = {"query", "mutating", "persisting", "locking", "signaling"}

# All tool classes that have catalog_metadata()
ALL_TOOL_CLASSES = [
    ListBankRecordsTool,
    AddBankRecordTool,
    UpdateBankRecordTool,
    RemoveBankRecordTool,
    FinalizeBankTool,
    CompileArtifactTool,
    FinalizeArtifactTool,
    CompleteWizardTool,
    RestartWizardTool,
    ListCatalogTool,
    SaveToCatalogTool,
    LoadFromCatalogTool,
]


def _make_bank(
    name: str = "ingredients",
    required: list[str] | None = None,
    match_fields: list[str] | None = None,
) -> MemoryBank:
    """Create a MemoryBank backed by SyncMemoryDatabase."""
    schema: dict[str, Any] = {}
    if required:
        schema["required"] = required
    return MemoryBank(
        name=name,
        schema=schema,
        db=SyncMemoryDatabase(),
        match_fields=match_fields,
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
    banks: dict[str, MemoryBank] | None = None,
    artifact: ArtifactBank | None = None,
    catalog: ArtifactBankCatalog | None = None,
    completion_signal: dict[str, Any] | None = None,
    restart_signal: dict[str, Any] | None = None,
) -> ToolExecutionContext:
    """Create a ToolExecutionContext with optional extras."""
    extra: dict[str, Any] = {}
    if banks is not None:
        extra["banks"] = banks
    if artifact is not None:
        extra["artifact"] = artifact
    if catalog is not None:
        extra["catalog"] = catalog
    if completion_signal is not None:
        extra["_completion_signal"] = completion_signal
    if restart_signal is not None:
        extra["_restart_signal"] = restart_signal
    return ToolExecutionContext(
        conversation_id="test-conv",
        user_id="test-user",
        extra=extra,
    )


class TestEffectsMetadata:
    """Every tool's catalog_metadata() must include a valid 'effects' tuple."""

    @pytest.mark.parametrize(
        "tool_class",
        ALL_TOOL_CLASSES,
        ids=[cls.__name__ for cls in ALL_TOOL_CLASSES],
    )
    def test_effects_key_present(self, tool_class: type) -> None:
        """catalog_metadata() includes 'effects' key."""
        metadata = tool_class.catalog_metadata()
        assert "effects" in metadata, (
            f"{tool_class.__name__}.catalog_metadata() missing 'effects' key"
        )

    @pytest.mark.parametrize(
        "tool_class",
        ALL_TOOL_CLASSES,
        ids=[cls.__name__ for cls in ALL_TOOL_CLASSES],
    )
    def test_effects_is_tuple(self, tool_class: type) -> None:
        """Effects value must be a tuple."""
        metadata = tool_class.catalog_metadata()
        effects = metadata["effects"]
        assert isinstance(effects, tuple), (
            f"{tool_class.__name__} effects is {type(effects).__name__}, "
            "expected tuple"
        )

    @pytest.mark.parametrize(
        "tool_class",
        ALL_TOOL_CLASSES,
        ids=[cls.__name__ for cls in ALL_TOOL_CLASSES],
    )
    def test_effects_values_valid(self, tool_class: type) -> None:
        """All effect values must be from the valid set."""
        metadata = tool_class.catalog_metadata()
        effects = metadata["effects"]
        for effect in effects:
            assert effect in VALID_EFFECTS, (
                f"{tool_class.__name__} has invalid effect '{effect}'. "
                f"Valid: {VALID_EFFECTS}"
            )

    def test_expected_effects_per_tool(self) -> None:
        """Verify specific effect classifications per 02b §5.3."""
        expected = {
            "list_bank_records": ("query",),
            "add_bank_record": ("mutating", "persisting"),
            "update_bank_record": ("mutating", "persisting"),
            "remove_bank_record": ("mutating", "persisting"),
            "finalize_bank": ("locking",),
            "compile_artifact": ("query",),
            "finalize_artifact": ("locking",),
            "complete_wizard": ("signaling",),
            "restart_wizard": ("signaling",),
            "list_catalog": ("query",),
            "save_to_catalog": ("persisting",),
            "load_from_catalog": ("mutating",),
        }
        for tool_class in ALL_TOOL_CLASSES:
            metadata = tool_class.catalog_metadata()
            name = metadata["name"]
            assert name in expected, f"Unexpected tool name: {name}"
            assert metadata["effects"] == expected[name], (
                f"{name}: expected effects {expected[name]}, "
                f"got {metadata['effects']}"
            )


class TestResponseFormatConsistency:
    """ListBankRecordsTool and ListCatalogTool must return success_response() format."""

    @pytest.mark.asyncio
    async def test_list_bank_records_has_success_key(self) -> None:
        """ListBankRecordsTool returns 'success': True alongside records."""
        bank = _make_bank(required=["name"])
        bank.add({"name": "flour", "amount": "2 cups"})
        context = _make_context(banks={"ingredients": bank})
        tool = ListBankRecordsTool()

        result = await tool.execute_with_context(
            context, bank_name="ingredients",
        )
        assert result["success"] is True
        assert result["count"] == 1
        assert result["records"][0]["name"] == "flour"
        assert result["bank_name"] == "ingredients"

    @pytest.mark.asyncio
    async def test_list_bank_records_empty_has_success_key(self) -> None:
        """Empty bank still returns success format."""
        bank = _make_bank()
        context = _make_context(banks={"ingredients": bank})
        tool = ListBankRecordsTool()

        result = await tool.execute_with_context(
            context, bank_name="ingredients",
        )
        assert result["success"] is True
        assert result["count"] == 0
        assert result["records"] == []

    @pytest.mark.asyncio
    async def test_list_catalog_has_success_key(self) -> None:
        """ListCatalogTool returns 'success': True alongside entries."""
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        artifact = _make_artifact(
            recipe_name="Cookies",
            ingredients=[{"name": "flour"}],
        )
        catalog.save(artifact)

        context = _make_context(catalog=catalog)
        tool = ListCatalogTool()

        result = await tool.execute_with_context(context)
        assert result["success"] is True
        assert result["count"] == 1
        assert result["entries"][0]["name"] == "recipe"

    @pytest.mark.asyncio
    async def test_list_catalog_empty_has_success_key(self) -> None:
        """Empty catalog still returns success format."""
        catalog = ArtifactBankCatalog(SyncMemoryDatabase())
        context = _make_context(catalog=catalog)
        tool = ListCatalogTool()

        result = await tool.execute_with_context(context)
        assert result["success"] is True
        assert result["count"] == 0
        assert result["entries"] == []


class TestCatalogMetadataStructure:
    """All catalog_metadata() methods must include required keys."""

    @pytest.mark.parametrize(
        "tool_class",
        ALL_TOOL_CLASSES,
        ids=[cls.__name__ for cls in ALL_TOOL_CLASSES],
    )
    def test_required_keys(self, tool_class: type) -> None:
        """catalog_metadata() must include name, description, tags, effects."""
        metadata = tool_class.catalog_metadata()
        for key in ("name", "description", "tags", "effects"):
            assert key in metadata, (
                f"{tool_class.__name__}.catalog_metadata() missing '{key}'"
            )

    @pytest.mark.parametrize(
        "tool_class",
        ALL_TOOL_CLASSES,
        ids=[cls.__name__ for cls in ALL_TOOL_CLASSES],
    )
    def test_name_is_string(self, tool_class: type) -> None:
        metadata = tool_class.catalog_metadata()
        assert isinstance(metadata["name"], str)

    @pytest.mark.parametrize(
        "tool_class",
        ALL_TOOL_CLASSES,
        ids=[cls.__name__ for cls in ALL_TOOL_CLASSES],
    )
    def test_tags_is_tuple(self, tool_class: type) -> None:
        metadata = tool_class.catalog_metadata()
        assert isinstance(metadata["tags"], tuple)
