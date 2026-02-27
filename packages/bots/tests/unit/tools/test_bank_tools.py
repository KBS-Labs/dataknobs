"""Tests for tools/bank_tools.py."""

from __future__ import annotations

from typing import Any

import pytest
from dataknobs_data.backends.memory import SyncMemoryDatabase
from dataknobs_llm.tools.context import ToolExecutionContext

from dataknobs_bots.memory.bank import MemoryBank
from dataknobs_bots.tools.bank_tools import (
    AddBankRecordTool,
    FinalizeBankTool,
    ListBankRecordsTool,
    RemoveBankRecordTool,
    UpdateBankRecordTool,
    _get_bank_from_context,
)


def _make_bank(
    name: str = "ingredients",
    required: list[str] | None = None,
) -> MemoryBank:
    """Create a MemoryBank backed by SyncMemoryDatabase."""
    schema: dict[str, Any] = {}
    if required:
        schema["required"] = required
    return MemoryBank(
        name=name,
        schema=schema,
        db=SyncMemoryDatabase(),
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
        context = _make_context()  # No banks
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


class TestListBankRecordsTool:
    """Tests for ListBankRecordsTool."""

    @pytest.mark.asyncio
    async def test_empty_bank(self) -> None:
        bank = _make_bank()
        context = _make_context(banks={"ingredients": bank})
        tool = ListBankRecordsTool(
            bank_name="ingredients", field_names=["name", "amount"]
        )

        result = await tool.execute_with_context(context)

        assert result["count"] == 0
        assert result["records"] == []
        assert result["bank_name"] == "ingredients"

    @pytest.mark.asyncio
    async def test_bank_with_records(self) -> None:
        bank = _make_bank(required=["name"])
        bank.add({"name": "flour", "amount": "2 cups"})
        bank.add({"name": "sugar", "amount": "1 cup"})
        context = _make_context(banks={"ingredients": bank})
        tool = ListBankRecordsTool(
            bank_name="ingredients", field_names=["name", "amount"]
        )

        result = await tool.execute_with_context(context)

        assert result["count"] == 2
        assert len(result["records"]) == 2
        names = [r["name"] for r in result["records"]]
        assert "flour" in names
        assert "sugar" in names
        # Check amounts are included
        flour_rec = next(r for r in result["records"] if r["name"] == "flour")
        assert flour_rec["amount"] == "2 cups"

    @pytest.mark.asyncio
    async def test_schema_has_no_required_properties(self) -> None:
        tool = ListBankRecordsTool(bank_name="ingredients")
        schema = tool.schema
        assert schema["type"] == "object"
        assert "required" not in schema

    @pytest.mark.asyncio
    async def test_no_banks_in_context(self) -> None:
        context = _make_context()  # No banks
        tool = ListBankRecordsTool(bank_name="ingredients")
        with pytest.raises(ValueError, match="banks in execution context"):
            await tool.execute_with_context(context)


class TestAddBankRecordTool:
    """Tests for AddBankRecordTool."""

    @pytest.mark.asyncio
    async def test_add_new_record(self) -> None:
        bank = _make_bank(required=["name"])
        context = _make_context(banks={"ingredients": bank})
        tool = AddBankRecordTool(
            bank_name="ingredients",
            field_names=["name", "amount"],
            required_fields=["name"],
            lookup_field="name",
        )

        result = await tool.execute_with_context(
            context, name="flour", amount="2 cups"
        )

        assert result["success"] is True
        assert result["data"] == {"name": "flour", "amount": "2 cups"}
        assert result["total_records"] == 1
        assert bank.count() == 1

    @pytest.mark.asyncio
    async def test_duplicate_detection(self) -> None:
        bank = _make_bank(required=["name"])
        bank.add({"name": "flour", "amount": "2 cups"})
        context = _make_context(banks={"ingredients": bank})
        tool = AddBankRecordTool(
            bank_name="ingredients",
            field_names=["name", "amount"],
            required_fields=["name"],
            lookup_field="name",
        )

        result = await tool.execute_with_context(
            context, name="flour", amount="3 cups"
        )

        assert result["success"] is False
        assert "already exists" in result["error"]
        assert "update_bank_record" in result["error"]
        assert result["existing_record"]["name"] == "flour"
        # Bank should still have just 1 record
        assert bank.count() == 1

    @pytest.mark.asyncio
    async def test_add_without_optional_field(self) -> None:
        bank = _make_bank(required=["name"])
        context = _make_context(banks={"ingredients": bank})
        tool = AddBankRecordTool(
            bank_name="ingredients",
            field_names=["name", "amount"],
            required_fields=["name"],
            lookup_field="name",
        )

        result = await tool.execute_with_context(context, name="salt")

        assert result["success"] is True
        assert result["data"] == {"name": "salt"}
        assert bank.count() == 1

    @pytest.mark.asyncio
    async def test_schema_includes_required_fields(self) -> None:
        tool = AddBankRecordTool(
            bank_name="ingredients",
            field_names=["name", "amount"],
            required_fields=["name"],
        )
        schema = tool.schema
        assert "name" in schema["properties"]
        assert "amount" in schema["properties"]
        assert schema["required"] == ["name"]


class TestUpdateBankRecordTool:
    """Tests for UpdateBankRecordTool."""

    @pytest.mark.asyncio
    async def test_update_by_lookup(self) -> None:
        bank = _make_bank(required=["name"])
        bank.add({"name": "flour", "amount": "2 cups"})
        context = _make_context(banks={"ingredients": bank})
        tool = UpdateBankRecordTool(
            bank_name="ingredients",
            field_names=["name", "amount"],
            lookup_field="name",
        )

        result = await tool.execute_with_context(
            context, name="flour", amount="3 cups"
        )

        assert result["success"] is True
        assert result["updated_data"]["amount"] == "3 cups"
        assert result["updated_data"]["name"] == "flour"
        # Verify in bank
        records = bank.all()
        assert len(records) == 1
        assert records[0].data["amount"] == "3 cups"

    @pytest.mark.asyncio
    async def test_record_not_found(self) -> None:
        bank = _make_bank(required=["name"])
        bank.add({"name": "flour", "amount": "2 cups"})
        context = _make_context(banks={"ingredients": bank})
        tool = UpdateBankRecordTool(
            bank_name="ingredients",
            field_names=["name", "amount"],
            lookup_field="name",
        )

        result = await tool.execute_with_context(
            context, name="sugar", amount="1 cup"
        )

        assert result["success"] is False
        assert "No record found" in result["error"]
        assert "flour" in result["available"]

    @pytest.mark.asyncio
    async def test_missing_lookup_field(self) -> None:
        bank = _make_bank()
        context = _make_context(banks={"ingredients": bank})
        tool = UpdateBankRecordTool(
            bank_name="ingredients",
            field_names=["name", "amount"],
            lookup_field="name",
        )

        result = await tool.execute_with_context(context, amount="1 cup")

        assert result["success"] is False
        assert "Missing required field" in result["error"]

    @pytest.mark.asyncio
    async def test_schema_requires_lookup_field(self) -> None:
        tool = UpdateBankRecordTool(
            bank_name="ingredients",
            field_names=["name", "amount"],
            lookup_field="name",
        )
        schema = tool.schema
        assert schema["required"] == ["name"]
        assert "name" in schema["properties"]
        assert "amount" in schema["properties"]


class TestRemoveBankRecordTool:
    """Tests for RemoveBankRecordTool."""

    @pytest.mark.asyncio
    async def test_remove_by_lookup(self) -> None:
        bank = _make_bank(required=["name"])
        bank.add({"name": "flour", "amount": "2 cups"})
        bank.add({"name": "sugar", "amount": "1 cup"})
        context = _make_context(banks={"ingredients": bank})
        tool = RemoveBankRecordTool(
            bank_name="ingredients", lookup_field="name"
        )

        result = await tool.execute_with_context(context, name="flour")

        assert result["success"] is True
        assert result["removed"]["name"] == "flour"
        assert result["remaining_records"] == 1
        assert bank.count() == 1
        # Verify flour is gone
        remaining = bank.all()
        assert remaining[0].data["name"] == "sugar"

    @pytest.mark.asyncio
    async def test_record_not_found(self) -> None:
        bank = _make_bank(required=["name"])
        bank.add({"name": "flour", "amount": "2 cups"})
        context = _make_context(banks={"ingredients": bank})
        tool = RemoveBankRecordTool(
            bank_name="ingredients", lookup_field="name"
        )

        result = await tool.execute_with_context(context, name="sugar")

        assert result["success"] is False
        assert "No record found" in result["error"]
        assert "flour" in result["available"]
        assert bank.count() == 1

    @pytest.mark.asyncio
    async def test_missing_lookup_field(self) -> None:
        bank = _make_bank()
        context = _make_context(banks={"ingredients": bank})
        tool = RemoveBankRecordTool(
            bank_name="ingredients", lookup_field="name"
        )

        result = await tool.execute_with_context(context)

        assert result["success"] is False
        assert "Missing required field" in result["error"]

    @pytest.mark.asyncio
    async def test_schema_requires_lookup_field(self) -> None:
        tool = RemoveBankRecordTool(
            bank_name="ingredients", lookup_field="name"
        )
        schema = tool.schema
        assert schema["required"] == ["name"]


class TestFinalizeBankTool:
    """Tests for FinalizeBankTool."""

    @pytest.mark.asyncio
    async def test_finalize_with_records(self) -> None:
        bank = _make_bank(required=["name"])
        bank.add({"name": "flour", "amount": "2 cups"})
        bank.add({"name": "sugar", "amount": "1 cup"})
        context = _make_context(banks={"ingredients": bank})
        tool = FinalizeBankTool(bank_name="ingredients")

        result = await tool.execute_with_context(context)

        assert result["success"] is True
        assert result["finalized"] is True
        assert result["record_count"] == 2
        assert result["bank_name"] == "ingredients"
        assert len(result["records"]) == 2

    @pytest.mark.asyncio
    async def test_finalize_empty_bank(self) -> None:
        bank = _make_bank()
        context = _make_context(banks={"ingredients": bank})
        tool = FinalizeBankTool(bank_name="ingredients")

        result = await tool.execute_with_context(context)

        assert result["success"] is True
        assert result["record_count"] == 0

    @pytest.mark.asyncio
    async def test_schema_has_no_required_properties(self) -> None:
        tool = FinalizeBankTool(bank_name="ingredients")
        schema = tool.schema
        assert schema["type"] == "object"
        assert "required" not in schema


class TestCatalogMetadata:
    """Tests for catalog_metadata classmethods."""

    def test_list_bank_records_metadata(self) -> None:
        meta = ListBankRecordsTool.catalog_metadata()
        assert meta["name"] == "list_bank_records"
        assert "tags" in meta

    def test_add_bank_record_metadata(self) -> None:
        meta = AddBankRecordTool.catalog_metadata()
        assert meta["name"] == "add_bank_record"
        assert "default_params" in meta

    def test_update_bank_record_metadata(self) -> None:
        meta = UpdateBankRecordTool.catalog_metadata()
        assert meta["name"] == "update_bank_record"

    def test_remove_bank_record_metadata(self) -> None:
        meta = RemoveBankRecordTool.catalog_metadata()
        assert meta["name"] == "remove_bank_record"

    def test_finalize_bank_metadata(self) -> None:
        meta = FinalizeBankTool.catalog_metadata()
        assert meta["name"] == "finalize_bank"
