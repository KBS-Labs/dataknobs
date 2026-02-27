"""Tests for tools/bank_tools.py."""

from __future__ import annotations

from typing import Any

import pytest
from dataknobs_data.backends.memory import SyncMemoryDatabase
from dataknobs_llm.tools.context import ToolExecutionContext

from dataknobs_bots.memory.bank import MemoryBank
from dataknobs_bots.memory.artifact_bank import ArtifactBank
from dataknobs_bots.tools.bank_tools import (
    AddBankRecordTool,
    CompileArtifactTool,
    FinalizeBankTool,
    ListBankRecordsTool,
    RemoveBankRecordTool,
    UpdateBankRecordTool,
    _get_bank_from_context,
    _resolve_lookup_field,
    _validate_record_id,
)


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


def _make_context(
    banks: dict[str, MemoryBank] | None = None,
) -> ToolExecutionContext:
    """Create a ToolExecutionContext with banks in extra."""
    extra: dict[str, Any] = {}
    if banks is not None:
        extra["banks"] = banks
    return ToolExecutionContext(
        conversation_id="test-conv",
        user_id="test-user",
        extra=extra,
    )


class TestGetBankFromContext:
    """Tests for _get_bank_from_context helper."""

    def test_missing_banks_raises(self) -> None:
        context = _make_context()
        with pytest.raises(ValueError, match="banks in execution context"):
            _get_bank_from_context(context, "ingredients")

    def test_missing_bank_name_raises(self) -> None:
        bank = _make_bank("other")
        context = _make_context(banks={"other": bank})
        with pytest.raises(ValueError, match="'ingredients' not found"):
            _get_bank_from_context(context, "ingredients")

    def test_returns_bank(self) -> None:
        bank = _make_bank("ingredients")
        context = _make_context(banks={"ingredients": bank})
        result = _get_bank_from_context(context, "ingredients")
        assert result is bank


class TestResolveLookupField:
    """Tests for _resolve_lookup_field helper."""

    def test_uses_match_fields(self) -> None:
        bank = _make_bank(match_fields=["name"])
        assert _resolve_lookup_field(bank, {"name": "x"}) == "name"

    def test_falls_back_to_required(self) -> None:
        bank = _make_bank(required=["instruction"])
        assert _resolve_lookup_field(bank, {"instruction": "x"}) == "instruction"

    def test_match_fields_takes_priority(self) -> None:
        bank = _make_bank(required=["a"], match_fields=["b"])
        assert _resolve_lookup_field(bank, {"a": "1", "b": "2"}) == "b"

    def test_returns_none_when_no_info(self) -> None:
        bank = _make_bank()
        assert _resolve_lookup_field(bank, {"x": "1"}) is None


class TestToolNameOverride:
    """Tests for the tool_name constructor parameter."""

    def test_list_custom_name(self) -> None:
        tool = ListBankRecordsTool(tool_name="list_items")
        assert tool.name == "list_items"

    def test_list_default_name(self) -> None:
        tool = ListBankRecordsTool()
        assert tool.name == "list_bank_records"

    def test_add_custom_name(self) -> None:
        tool = AddBankRecordTool(tool_name="add_item")
        assert tool.name == "add_item"

    def test_add_default_name(self) -> None:
        tool = AddBankRecordTool()
        assert tool.name == "add_bank_record"

    def test_update_custom_name(self) -> None:
        tool = UpdateBankRecordTool(tool_name="update_item")
        assert tool.name == "update_item"

    def test_remove_custom_name(self) -> None:
        tool = RemoveBankRecordTool(tool_name="remove_item")
        assert tool.name == "remove_item"

    def test_finalize_custom_name(self) -> None:
        tool = FinalizeBankTool(tool_name="finalize_items")
        assert tool.name == "finalize_items"


class TestListBankRecordsTool:
    """Tests for ListBankRecordsTool."""

    @pytest.mark.asyncio
    async def test_empty_bank(self) -> None:
        bank = _make_bank()
        context = _make_context(banks={"ingredients": bank})
        tool = ListBankRecordsTool()

        result = await tool.execute_with_context(
            context, bank_name="ingredients"
        )

        assert result["count"] == 0
        assert result["records"] == []
        assert result["bank_name"] == "ingredients"

    @pytest.mark.asyncio
    async def test_bank_with_records(self) -> None:
        bank = _make_bank(required=["name"])
        bank.add({"name": "flour", "amount": "2 cups"})
        bank.add({"name": "sugar", "amount": "1 cup"})
        context = _make_context(banks={"ingredients": bank})
        tool = ListBankRecordsTool()

        result = await tool.execute_with_context(
            context, bank_name="ingredients"
        )

        assert result["count"] == 2
        assert len(result["records"]) == 2
        names = [r["name"] for r in result["records"]]
        assert "flour" in names
        assert "sugar" in names
        # All data fields are included
        flour_rec = next(r for r in result["records"] if r["name"] == "flour")
        assert flour_rec["amount"] == "2 cups"

    @pytest.mark.asyncio
    async def test_missing_bank_name(self) -> None:
        bank = _make_bank()
        context = _make_context(banks={"ingredients": bank})
        tool = ListBankRecordsTool()

        result = await tool.execute_with_context(context)

        assert result["success"] is False
        assert "bank_name" in result["error"]

    @pytest.mark.asyncio
    async def test_schema_requires_bank_name(self) -> None:
        tool = ListBankRecordsTool()
        schema = tool.schema
        assert schema["type"] == "object"
        assert "bank_name" in schema["properties"]
        assert schema["required"] == ["bank_name"]

    @pytest.mark.asyncio
    async def test_no_banks_in_context(self) -> None:
        context = _make_context()
        tool = ListBankRecordsTool()
        with pytest.raises(ValueError, match="banks in execution context"):
            await tool.execute_with_context(context, bank_name="ingredients")


class TestAddBankRecordTool:
    """Tests for AddBankRecordTool."""

    @pytest.mark.asyncio
    async def test_add_new_record(self) -> None:
        bank = _make_bank(required=["name"], match_fields=["name"])
        context = _make_context(banks={"ingredients": bank})
        tool = AddBankRecordTool()

        result = await tool.execute_with_context(
            context,
            bank_name="ingredients",
            data={"name": "flour", "amount": "2 cups"},
        )

        assert result["success"] is True
        assert result["data"] == {"name": "flour", "amount": "2 cups"}
        assert result["total_records"] == 1
        assert bank.count() == 1

    @pytest.mark.asyncio
    async def test_duplicate_detection(self) -> None:
        bank = _make_bank(required=["name"], match_fields=["name"])
        bank.add({"name": "flour", "amount": "2 cups"})
        context = _make_context(banks={"ingredients": bank})
        tool = AddBankRecordTool()

        result = await tool.execute_with_context(
            context,
            bank_name="ingredients",
            data={"name": "flour", "amount": "3 cups"},
        )

        assert result["success"] is False
        assert "already exists" in result["error"]
        assert "update tool" in result["error"]
        assert result["existing_record"]["name"] == "flour"
        assert bank.count() == 1

    @pytest.mark.asyncio
    async def test_add_without_optional_field(self) -> None:
        bank = _make_bank(required=["name"], match_fields=["name"])
        context = _make_context(banks={"ingredients": bank})
        tool = AddBankRecordTool()

        result = await tool.execute_with_context(
            context,
            bank_name="ingredients",
            data={"name": "salt"},
        )

        assert result["success"] is True
        assert result["data"] == {"name": "salt"}
        assert bank.count() == 1

    @pytest.mark.asyncio
    async def test_missing_bank_name(self) -> None:
        tool = AddBankRecordTool()
        bank = _make_bank()
        context = _make_context(banks={"ingredients": bank})

        result = await tool.execute_with_context(
            context, data={"name": "flour"}
        )

        assert result["success"] is False
        assert "bank_name" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_data(self) -> None:
        tool = AddBankRecordTool()
        bank = _make_bank()
        context = _make_context(banks={"ingredients": bank})

        result = await tool.execute_with_context(
            context, bank_name="ingredients"
        )

        assert result["success"] is False
        assert "data" in result["error"]

    @pytest.mark.asyncio
    async def test_schema_requires_bank_name_and_data(self) -> None:
        tool = AddBankRecordTool()
        schema = tool.schema
        assert "bank_name" in schema["properties"]
        assert "data" in schema["properties"]
        assert set(schema["required"]) == {"bank_name", "data"}


class TestUpdateBankRecordTool:
    """Tests for UpdateBankRecordTool."""

    @pytest.mark.asyncio
    async def test_update_by_record_id(self) -> None:
        bank = _make_bank(required=["name"], match_fields=["name"])
        rec_id = bank.add({"name": "flour", "amount": "2 cups"})
        context = _make_context(banks={"ingredients": bank})
        tool = UpdateBankRecordTool()

        result = await tool.execute_with_context(
            context,
            bank_name="ingredients",
            record_id=rec_id,
            data={"amount": "3 cups"},
        )

        assert result["success"] is True
        assert result["record_id"] == rec_id
        assert result["updated_data"]["amount"] == "3 cups"
        assert result["updated_data"]["name"] == "flour"
        records = bank.all()
        assert len(records) == 1
        assert records[0].data["amount"] == "3 cups"

    @pytest.mark.asyncio
    async def test_update_single_field_record(self) -> None:
        """Updating the only field (e.g. instruction) works via record_id."""
        bank = _make_bank(
            "instructions", required=["instruction"],
        )
        rec_id = bank.add({"instruction": "Mix dry and wet ingredients"})
        context = _make_context(banks={"instructions": bank})
        tool = UpdateBankRecordTool()

        result = await tool.execute_with_context(
            context,
            bank_name="instructions",
            record_id=rec_id,
            data={"instruction": "Mix dry and wet ingredients separately"},
        )

        assert result["success"] is True
        assert result["updated_data"]["instruction"] == (
            "Mix dry and wet ingredients separately"
        )
        records = bank.all()
        assert records[0].data["instruction"] == (
            "Mix dry and wet ingredients separately"
        )

    @pytest.mark.asyncio
    async def test_record_not_found(self) -> None:
        bank = _make_bank(required=["name"], match_fields=["name"])
        bank.add({"name": "flour", "amount": "2 cups"})
        context = _make_context(banks={"ingredients": bank})
        tool = UpdateBankRecordTool()

        result = await tool.execute_with_context(
            context,
            bank_name="ingredients",
            record_id="deadbeef0000",
            data={"amount": "1 cup"},
        )

        assert result["success"] is False
        assert "No record found" in result["error"]
        assert "list_bank_records" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_record_id(self) -> None:
        bank = _make_bank(required=["name"])
        context = _make_context(banks={"ingredients": bank})
        tool = UpdateBankRecordTool()

        result = await tool.execute_with_context(
            context,
            bank_name="ingredients",
            data={"amount": "1 cup"},
        )

        assert result["success"] is False
        assert "record_id" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_data(self) -> None:
        bank = _make_bank(required=["name"])
        context = _make_context(banks={"ingredients": bank})
        tool = UpdateBankRecordTool()

        result = await tool.execute_with_context(
            context,
            bank_name="ingredients",
            record_id="deadbeef0001",
        )

        assert result["success"] is False
        assert "data" in result["error"]

    @pytest.mark.asyncio
    async def test_schema_requires_bank_name_record_id_and_data(self) -> None:
        tool = UpdateBankRecordTool()
        schema = tool.schema
        assert set(schema["required"]) == {"bank_name", "record_id", "data"}
        assert "record_id" in schema["properties"]


class TestRemoveBankRecordTool:
    """Tests for RemoveBankRecordTool."""

    @pytest.mark.asyncio
    async def test_remove_by_record_id(self) -> None:
        bank = _make_bank(required=["name"], match_fields=["name"])
        flour_id = bank.add({"name": "flour", "amount": "2 cups"})
        bank.add({"name": "sugar", "amount": "1 cup"})
        context = _make_context(banks={"ingredients": bank})
        tool = RemoveBankRecordTool()

        result = await tool.execute_with_context(
            context,
            bank_name="ingredients",
            record_id=flour_id,
        )

        assert result["success"] is True
        assert result["removed"]["name"] == "flour"
        assert result["removed"]["record_id"] == flour_id
        assert result["remaining_records"] == 1
        assert bank.count() == 1
        remaining = bank.all()
        assert remaining[0].data["name"] == "sugar"

    @pytest.mark.asyncio
    async def test_record_not_found(self) -> None:
        bank = _make_bank(required=["name"], match_fields=["name"])
        bank.add({"name": "flour", "amount": "2 cups"})
        context = _make_context(banks={"ingredients": bank})
        tool = RemoveBankRecordTool()

        result = await tool.execute_with_context(
            context,
            bank_name="ingredients",
            record_id="deadbeef0000",
        )

        assert result["success"] is False
        assert "No record found" in result["error"]
        assert "list_bank_records" in result["error"]
        assert bank.count() == 1

    @pytest.mark.asyncio
    async def test_missing_record_id(self) -> None:
        bank = _make_bank(required=["name"], match_fields=["name"])
        context = _make_context(banks={"ingredients": bank})
        tool = RemoveBankRecordTool()

        result = await tool.execute_with_context(
            context,
            bank_name="ingredients",
        )

        assert result["success"] is False
        assert "record_id" in result["error"]

    @pytest.mark.asyncio
    async def test_schema_requires_bank_name_and_record_id(self) -> None:
        tool = RemoveBankRecordTool()
        schema = tool.schema
        assert set(schema["required"]) == {"bank_name", "record_id"}
        assert "record_id" in schema["properties"]
        assert "data" not in schema["properties"]


class TestFinalizeBankTool:
    """Tests for FinalizeBankTool."""

    @pytest.mark.asyncio
    async def test_finalize_with_records(self) -> None:
        bank = _make_bank(required=["name"])
        bank.add({"name": "flour", "amount": "2 cups"})
        bank.add({"name": "sugar", "amount": "1 cup"})
        context = _make_context(banks={"ingredients": bank})
        tool = FinalizeBankTool()

        result = await tool.execute_with_context(
            context, bank_name="ingredients"
        )

        assert result["success"] is True
        assert result["finalized"] is True
        assert result["record_count"] == 2
        assert result["bank_name"] == "ingredients"
        assert len(result["records"]) == 2

    @pytest.mark.asyncio
    async def test_finalize_empty_bank(self) -> None:
        bank = _make_bank()
        context = _make_context(banks={"ingredients": bank})
        tool = FinalizeBankTool()

        result = await tool.execute_with_context(
            context, bank_name="ingredients"
        )

        assert result["success"] is True
        assert result["record_count"] == 0

    @pytest.mark.asyncio
    async def test_missing_bank_name(self) -> None:
        bank = _make_bank()
        context = _make_context(banks={"ingredients": bank})
        tool = FinalizeBankTool()

        result = await tool.execute_with_context(context)

        assert result["success"] is False
        assert "bank_name" in result["error"]

    @pytest.mark.asyncio
    async def test_schema_requires_bank_name(self) -> None:
        tool = FinalizeBankTool()
        schema = tool.schema
        assert schema["type"] == "object"
        assert "bank_name" in schema["properties"]
        assert schema["required"] == ["bank_name"]


class TestMultiBankContext:
    """Tests for tools operating across multiple banks."""

    @pytest.mark.asyncio
    async def test_add_to_different_banks(self) -> None:
        ing_bank = _make_bank("ingredients", required=["name"], match_fields=["name"])
        inst_bank = _make_bank(
            "instructions", required=["instruction"], match_fields=["instruction"]
        )
        context = _make_context(
            banks={"ingredients": ing_bank, "instructions": inst_bank}
        )
        tool = AddBankRecordTool()

        r1 = await tool.execute_with_context(
            context,
            bank_name="ingredients",
            data={"name": "flour", "amount": "2 cups"},
        )
        r2 = await tool.execute_with_context(
            context,
            bank_name="instructions",
            data={"instruction": "Preheat oven to 375F"},
        )

        assert r1["success"] is True
        assert r2["success"] is True
        assert ing_bank.count() == 1
        assert inst_bank.count() == 1

    @pytest.mark.asyncio
    async def test_list_from_different_banks(self) -> None:
        ing_bank = _make_bank("ingredients", required=["name"])
        ing_bank.add({"name": "flour"})
        inst_bank = _make_bank("instructions", required=["instruction"])
        inst_bank.add({"instruction": "Mix dry ingredients"})
        context = _make_context(
            banks={"ingredients": ing_bank, "instructions": inst_bank}
        )
        tool = ListBankRecordsTool()

        r1 = await tool.execute_with_context(
            context, bank_name="ingredients"
        )
        r2 = await tool.execute_with_context(
            context, bank_name="instructions"
        )

        assert r1["count"] == 1
        assert r1["records"][0]["name"] == "flour"
        assert r2["count"] == 1
        assert r2["records"][0]["instruction"] == "Mix dry ingredients"

    @pytest.mark.asyncio
    async def test_finalize_different_banks(self) -> None:
        ing_bank = _make_bank("ingredients", required=["name"])
        ing_bank.add({"name": "flour"})
        inst_bank = _make_bank("instructions", required=["instruction"])
        inst_bank.add({"instruction": "Step 1"})
        inst_bank.add({"instruction": "Step 2"})
        context = _make_context(
            banks={"ingredients": ing_bank, "instructions": inst_bank}
        )
        tool = FinalizeBankTool()

        r1 = await tool.execute_with_context(
            context, bank_name="ingredients"
        )
        r2 = await tool.execute_with_context(
            context, bank_name="instructions"
        )

        assert r1["record_count"] == 1
        assert r2["record_count"] == 2


class TestCatalogMetadata:
    """Tests for catalog_metadata classmethods."""

    def test_list_bank_records_metadata(self) -> None:
        meta = ListBankRecordsTool.catalog_metadata()
        assert meta["name"] == "list_bank_records"
        assert "tags" in meta

    def test_add_bank_record_metadata(self) -> None:
        meta = AddBankRecordTool.catalog_metadata()
        assert meta["name"] == "add_bank_record"

    def test_update_bank_record_metadata(self) -> None:
        meta = UpdateBankRecordTool.catalog_metadata()
        assert meta["name"] == "update_bank_record"

    def test_remove_bank_record_metadata(self) -> None:
        meta = RemoveBankRecordTool.catalog_metadata()
        assert meta["name"] == "remove_bank_record"

    def test_finalize_bank_metadata(self) -> None:
        meta = FinalizeBankTool.catalog_metadata()
        assert meta["name"] == "finalize_bank"


class TestValidateRecordId:
    """Tests for record_id format validation."""

    def test_valid_12_char_hex(self) -> None:
        assert _validate_record_id("a1b2c3d4e5f6") is None

    def test_valid_all_zeros(self) -> None:
        assert _validate_record_id("000000000000") is None

    def test_valid_all_f(self) -> None:
        assert _validate_record_id("ffffffffffff") is None

    def test_rejects_short_id(self) -> None:
        result = _validate_record_id("1")
        assert result is not None
        assert result["success"] is False
        assert "Invalid record_id format" in result["error"]
        assert "list_bank_records" in result["error"]

    def test_rejects_long_id(self) -> None:
        result = _validate_record_id("a1b2c3d4e5f6a")
        assert result is not None
        assert result["success"] is False

    def test_rejects_non_hex_chars(self) -> None:
        result = _validate_record_id("a1b2c3d4e5gz")
        assert result is not None
        assert result["success"] is False

    def test_rejects_uppercase_hex(self) -> None:
        result = _validate_record_id("A1B2C3D4E5F6")
        assert result is not None
        assert result["success"] is False

    def test_rejects_descriptive_string(self) -> None:
        result = _validate_record_id("nonexistent-id")
        assert result is not None
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_update_rejects_invalid_format(self) -> None:
        bank = _make_bank()
        context = _make_context(banks={"ingredients": bank})
        tool = UpdateBankRecordTool()

        result = await tool.execute_with_context(
            context,
            bank_name="ingredients",
            record_id="1",
            data={"amount": "2 cups"},
        )

        assert result["success"] is False
        assert "Invalid record_id format" in result["error"]

    @pytest.mark.asyncio
    async def test_remove_rejects_invalid_format(self) -> None:
        bank = _make_bank()
        context = _make_context(banks={"ingredients": bank})
        tool = RemoveBankRecordTool()

        result = await tool.execute_with_context(
            context,
            bank_name="ingredients",
            record_id="1",
        )

        assert result["success"] is False
        assert "Invalid record_id format" in result["error"]


def _make_artifact_context(
    with_data: bool = False,
) -> tuple[ToolExecutionContext, ArtifactBank]:
    """Create a context with an ArtifactBank for testing."""
    ingredients = _make_bank("ingredients", required=["name"], match_fields=["name"])
    instructions = _make_bank("instructions", required=["instruction"])
    artifact = ArtifactBank(
        name="recipe",
        field_defs={"recipe_name": {"required": True}},
        sections={"ingredients": ingredients, "instructions": instructions},
    )
    if with_data:
        artifact.set_field("recipe_name", "Chocolate Chip Cookies")
        ingredients.add({"name": "flour", "amount": "2 cups"})
        instructions.add({"instruction": "Preheat oven to 375F"})

    extra: dict[str, Any] = {"artifact": artifact}
    context = ToolExecutionContext(
        conversation_id="test-conv",
        user_id="test-user",
        extra=extra,
    )
    return context, artifact


class TestCompileArtifactTool:
    """Tests for CompileArtifactTool."""

    @pytest.mark.asyncio
    async def test_no_artifact_in_context(self) -> None:
        context = _make_context()
        tool = CompileArtifactTool()

        result = await tool.execute_with_context(context)

        assert result["success"] is False
        assert "No artifact configured" in result["error"]

    @pytest.mark.asyncio
    async def test_validation_errors_returned(self) -> None:
        context, _artifact = _make_artifact_context(with_data=False)
        tool = CompileArtifactTool()

        result = await tool.execute_with_context(context)

        assert result["success"] is False
        assert "errors" in result
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_successful_compile(self) -> None:
        context, _artifact = _make_artifact_context(with_data=True)
        tool = CompileArtifactTool()

        result = await tool.execute_with_context(context)

        assert result["success"] is True
        assert "artifact" in result
        compiled = result["artifact"]
        assert compiled["recipe_name"] == "Chocolate Chip Cookies"
        assert len(compiled["ingredients"]) == 1
        assert len(compiled["instructions"]) == 1

    @pytest.mark.asyncio
    async def test_schema_no_required_params(self) -> None:
        tool = CompileArtifactTool()
        schema = tool.schema
        assert schema["type"] == "object"
        assert "required" not in schema

    def test_catalog_metadata(self) -> None:
        meta = CompileArtifactTool.catalog_metadata()
        assert meta["name"] == "compile_artifact"
        assert "artifact" in meta["tags"]

    def test_custom_tool_name(self) -> None:
        tool = CompileArtifactTool(tool_name="compile_recipe")
        assert tool.name == "compile_recipe"

    def test_default_tool_name(self) -> None:
        tool = CompileArtifactTool()
        assert tool.name == "compile_artifact"
