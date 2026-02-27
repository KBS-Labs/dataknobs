"""MemoryBank CRUD tools for wizard review stages.

Provides LLM-callable tools that operate on ``MemoryBank`` instances
injected into the tool execution context via ``context.extra["banks"]``.

Tools are parameterized at construction time with ``bank_name``,
``field_names``, ``lookup_field``, etc., so the same tool classes work
for any bank schema (ingredients, contacts, configuration items).

Tools:
- ListBankRecordsTool: List all records in a bank
- AddBankRecordTool: Add a record with duplicate detection
- UpdateBankRecordTool: Update a record by lookup field
- RemoveBankRecordTool: Remove a record by lookup field
- FinalizeBankTool: Confirm save (bank data already persisted)

Example:
    ```python
    from dataknobs_bots.tools.bank_tools import (
        ListBankRecordsTool, AddBankRecordTool, FinalizeBankTool,
    )

    list_tool = ListBankRecordsTool(
        bank_name="ingredients",
        field_names=["name", "amount"],
    )
    add_tool = AddBankRecordTool(
        bank_name="ingredients",
        field_names=["name", "amount"],
        required_fields=["name"],
        lookup_field="name",
    )
    ```
"""

from __future__ import annotations

import logging
from typing import Any

from dataknobs_llm.tools.context import ToolExecutionContext
from dataknobs_llm.tools.context_aware import ContextAwareTool

logger = logging.getLogger(__name__)


def _get_bank_from_context(context: ToolExecutionContext, bank_name: str) -> Any:
    """Extract a MemoryBank from the tool execution context.

    Args:
        context: Tool execution context with ``extra["banks"]``.
        bank_name: Name of the bank to retrieve.

    Returns:
        The ``MemoryBank`` instance.

    Raises:
        ValueError: If banks are not in context or the named bank is missing.
    """
    banks = context.extra.get("banks")
    if not banks:
        raise ValueError(
            "MemoryBank tools require banks in execution context. "
            "Ensure the wizard stage has banks configured."
        )
    bank = banks.get(bank_name)
    if bank is None:
        raise ValueError(
            f"Bank '{bank_name}' not found in context. "
            f"Available banks: {list(banks.keys())}"
        )
    return bank


class ListBankRecordsTool(ContextAwareTool):
    """Tool for listing all records in a MemoryBank.

    Returns all records with their field values and a total count.
    Used by the LLM to display the current state of the bank.

    Attributes:
        _bank_name: Name of the bank to list from.
        _field_names: Fields to include in the listing.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "list_bank_records",
            "description": (
                "List all records in a memory bank."
            ),
            "default_params": {
                "bank_name": "items",
                "field_names": ["name"],
            },
            "tags": ("wizard", "bank"),
        }

    def __init__(
        self,
        bank_name: str = "items",
        field_names: list[str] | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            bank_name: Name of the bank to list from.
            field_names: Fields to include in the record listing.
        """
        super().__init__(
            name="list_bank_records",
            description=(
                f"List all records in the '{bank_name}' bank "
                f"with their current values."
            ),
        )
        self._bank_name = bank_name
        self._field_names = field_names or ["name"]

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {},
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """List all records in the bank.

        Args:
            context: Execution context with banks.

        Returns:
            Dict with records list and count.
        """
        bank = _get_bank_from_context(context, self._bank_name)
        records = bank.all()

        items = []
        for record in records:
            item: dict[str, Any] = {"record_id": record.record_id}
            for field_name in self._field_names:
                item[field_name] = record.data.get(field_name)
            items.append(item)

        logger.debug(
            "Listed %d records from bank '%s'",
            len(items),
            self._bank_name,
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "records": items,
            "count": len(items),
            "bank_name": self._bank_name,
        }


class AddBankRecordTool(ContextAwareTool):
    """Tool for adding a record to a MemoryBank with duplicate detection.

    Pre-checks for duplicates via the lookup field before adding.
    Returns a helpful error message if a duplicate exists, suggesting
    the update tool instead.

    Attributes:
        _bank_name: Name of the bank to add to.
        _field_names: All field names for the record schema.
        _required_fields: Fields that must be provided.
        _lookup_field: Field used for duplicate detection.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "add_bank_record",
            "description": (
                "Add a new record to a memory bank."
            ),
            "default_params": {
                "bank_name": "items",
                "field_names": ["name"],
                "required_fields": ["name"],
                "lookup_field": "name",
            },
            "tags": ("wizard", "bank"),
        }

    def __init__(
        self,
        bank_name: str = "items",
        field_names: list[str] | None = None,
        required_fields: list[str] | None = None,
        lookup_field: str = "name",
    ) -> None:
        """Initialize the tool.

        Args:
            bank_name: Name of the bank to add to.
            field_names: All field names accepted by this tool.
            required_fields: Fields that must be provided.
            lookup_field: Field used for duplicate detection.
        """
        super().__init__(
            name="add_bank_record",
            description=(
                f"Add a new record to the '{bank_name}' bank. "
                f"Checks for duplicates by '{lookup_field}'."
            ),
        )
        self._bank_name = bank_name
        self._field_names = field_names or ["name"]
        self._required_fields = required_fields or ["name"]
        self._lookup_field = lookup_field

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        properties: dict[str, Any] = {}
        for field_name in self._field_names:
            properties[field_name] = {
                "type": "string",
                "description": f"The {field_name} of the record",
            }
        return {
            "type": "object",
            "properties": properties,
            "required": list(self._required_fields),
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add a record to the bank.

        Args:
            context: Execution context with banks.
            **kwargs: Field values from the LLM.

        Returns:
            Dict with add result or duplicate error.
        """
        bank = _get_bank_from_context(context, self._bank_name)

        # Build data dict from known field names
        data: dict[str, Any] = {}
        for field_name in self._field_names:
            if field_name in kwargs:
                data[field_name] = kwargs[field_name]

        # Check for duplicates by lookup field
        lookup_value = data.get(self._lookup_field)
        if lookup_value is not None:
            duplicates = bank.find(**{self._lookup_field: lookup_value})
            if duplicates:
                existing = duplicates[0]
                logger.debug(
                    "Duplicate detected in bank '%s': %s='%s' (record %s)",
                    self._bank_name,
                    self._lookup_field,
                    lookup_value,
                    existing.record_id,
                    extra={"conversation_id": context.conversation_id},
                )
                return {
                    "success": False,
                    "error": (
                        f"A record with {self._lookup_field}="
                        f"'{lookup_value}' already exists. "
                        f"Use update_bank_record to modify it."
                    ),
                    "existing_record": {
                        "record_id": existing.record_id,
                        **{f: existing.data.get(f) for f in self._field_names},
                    },
                }

        record_id = bank.add(data)

        logger.debug(
            "Added record %s to bank '%s'",
            record_id,
            self._bank_name,
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "success": True,
            "record_id": record_id,
            "data": data,
            "total_records": bank.count(),
        }


class UpdateBankRecordTool(ContextAwareTool):
    """Tool for updating a record in a MemoryBank by lookup field.

    Finds the record via the lookup field value, then merges new
    values into the existing record data.

    Attributes:
        _bank_name: Name of the bank to update in.
        _field_names: All field names for the record schema.
        _lookup_field: Field used to find the record.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "update_bank_record",
            "description": (
                "Update an existing record in a memory bank."
            ),
            "default_params": {
                "bank_name": "items",
                "field_names": ["name"],
                "lookup_field": "name",
            },
            "tags": ("wizard", "bank"),
        }

    def __init__(
        self,
        bank_name: str = "items",
        field_names: list[str] | None = None,
        lookup_field: str = "name",
    ) -> None:
        """Initialize the tool.

        Args:
            bank_name: Name of the bank to update in.
            field_names: All field names for the record schema.
            lookup_field: Field used to find the record to update.
        """
        super().__init__(
            name="update_bank_record",
            description=(
                f"Update an existing record in the '{bank_name}' bank "
                f"by looking it up with '{lookup_field}'."
            ),
        )
        self._bank_name = bank_name
        self._field_names = field_names or ["name"]
        self._lookup_field = lookup_field

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        properties: dict[str, Any] = {
            self._lookup_field: {
                "type": "string",
                "description": (
                    f"The {self._lookup_field} of the record to update"
                ),
            },
        }
        for field_name in self._field_names:
            if field_name != self._lookup_field:
                properties[field_name] = {
                    "type": "string",
                    "description": f"New value for {field_name}",
                }
        return {
            "type": "object",
            "properties": properties,
            "required": [self._lookup_field],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update a record in the bank.

        Args:
            context: Execution context with banks.
            **kwargs: Lookup field value + new field values from the LLM.

        Returns:
            Dict with update result or not-found error.
        """
        bank = _get_bank_from_context(context, self._bank_name)

        lookup_value = kwargs.get(self._lookup_field)
        if lookup_value is None:
            return {
                "success": False,
                "error": f"Missing required field: {self._lookup_field}",
            }

        matches = bank.find(**{self._lookup_field: lookup_value})
        if not matches:
            # List available records to help the LLM
            all_records = bank.all()
            available = [
                r.data.get(self._lookup_field) for r in all_records
            ]
            return {
                "success": False,
                "error": (
                    f"No record found with {self._lookup_field}="
                    f"'{lookup_value}'."
                ),
                "available": available,
            }

        record = matches[0]
        # Merge new values into existing data
        updated_data = dict(record.data)
        for field_name in self._field_names:
            if field_name in kwargs:
                updated_data[field_name] = kwargs[field_name]

        bank.update(record.record_id, updated_data)

        logger.debug(
            "Updated record %s in bank '%s' (lookup: %s='%s')",
            record.record_id,
            self._bank_name,
            self._lookup_field,
            lookup_value,
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "success": True,
            "record_id": record.record_id,
            "updated_data": updated_data,
        }


class RemoveBankRecordTool(ContextAwareTool):
    """Tool for removing a record from a MemoryBank by lookup field.

    Finds the record via the lookup field value and removes it.

    Attributes:
        _bank_name: Name of the bank to remove from.
        _lookup_field: Field used to find the record.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "remove_bank_record",
            "description": (
                "Remove a record from a memory bank."
            ),
            "default_params": {
                "bank_name": "items",
                "lookup_field": "name",
            },
            "tags": ("wizard", "bank"),
        }

    def __init__(
        self,
        bank_name: str = "items",
        lookup_field: str = "name",
    ) -> None:
        """Initialize the tool.

        Args:
            bank_name: Name of the bank to remove from.
            lookup_field: Field used to find the record to remove.
        """
        super().__init__(
            name="remove_bank_record",
            description=(
                f"Remove a record from the '{bank_name}' bank "
                f"by looking it up with '{lookup_field}'."
            ),
        )
        self._bank_name = bank_name
        self._lookup_field = lookup_field

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                self._lookup_field: {
                    "type": "string",
                    "description": (
                        f"The {self._lookup_field} of the record to remove"
                    ),
                },
            },
            "required": [self._lookup_field],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Remove a record from the bank.

        Args:
            context: Execution context with banks.
            **kwargs: Lookup field value from the LLM.

        Returns:
            Dict with removal result or not-found error.
        """
        bank = _get_bank_from_context(context, self._bank_name)

        lookup_value = kwargs.get(self._lookup_field)
        if lookup_value is None:
            return {
                "success": False,
                "error": f"Missing required field: {self._lookup_field}",
            }

        matches = bank.find(**{self._lookup_field: lookup_value})
        if not matches:
            all_records = bank.all()
            available = [
                r.data.get(self._lookup_field) for r in all_records
            ]
            return {
                "success": False,
                "error": (
                    f"No record found with {self._lookup_field}="
                    f"'{lookup_value}'."
                ),
                "available": available,
            }

        record = matches[0]
        bank.remove(record.record_id)

        logger.debug(
            "Removed record %s from bank '%s' (lookup: %s='%s')",
            record.record_id,
            self._bank_name,
            self._lookup_field,
            lookup_value,
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "success": True,
            "removed": {
                "record_id": record.record_id,
                **{f: record.data.get(f) for f in record.data},
            },
            "remaining_records": bank.count(),
        }


class FinalizeBankTool(ContextAwareTool):
    """Tool for confirming the bank contents are finalized.

    This is a declarative confirmation tool. Bank data is already
    persisted in conversation metadata automatically by the wizard's
    ``_save_wizard_state``. This tool simply returns a confirmation
    with the final record count so the LLM can present it to the user.

    Attributes:
        _bank_name: Name of the bank to finalize.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "finalize_bank",
            "description": (
                "Confirm and finalize the memory bank contents."
            ),
            "default_params": {
                "bank_name": "items",
            },
            "tags": ("wizard", "bank"),
        }

    def __init__(
        self,
        bank_name: str = "items",
    ) -> None:
        """Initialize the tool.

        Args:
            bank_name: Name of the bank to finalize.
        """
        super().__init__(
            name="finalize_bank",
            description=(
                f"Confirm and finalize the '{bank_name}' bank contents. "
                f"Data is already saved automatically."
            ),
        )
        self._bank_name = bank_name

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {},
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Finalize the bank contents.

        Args:
            context: Execution context with banks.

        Returns:
            Dict with confirmation and record count.
        """
        bank = _get_bank_from_context(context, self._bank_name)
        count = bank.count()
        records = bank.all()

        items = []
        for record in records:
            items.append(dict(record.data))

        logger.info(
            "Finalized bank '%s' with %d records",
            self._bank_name,
            count,
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "success": True,
            "finalized": True,
            "bank_name": self._bank_name,
            "record_count": count,
            "records": items,
        }
