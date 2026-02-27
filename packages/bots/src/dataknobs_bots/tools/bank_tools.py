"""MemoryBank CRUD tools and ArtifactBank tools for wizard review stages.

Provides LLM-callable tools that operate on ``MemoryBank`` instances
injected into the tool execution context via ``context.extra["banks"]``.

All tools accept ``bank_name`` and ``data`` as runtime parameters so a
single tool instance can operate on any bank in the context.  The LLM
specifies which bank to target and passes record data as a dict.

Tools:
- ListBankRecordsTool: List all records in a bank (returns record_id per record)
- AddBankRecordTool: Add a record with duplicate detection
- UpdateBankRecordTool: Update a record by record_id
- RemoveBankRecordTool: Remove a record by record_id
- FinalizeBankTool: Confirm save (bank data already persisted)
- CompileArtifactTool: Compile all artifact fields and sections into output

Example:
    ```python
    from dataknobs_bots.tools.bank_tools import (
        ListBankRecordsTool, AddBankRecordTool, FinalizeBankTool,
    )

    # One instance per tool type — works with any bank at runtime.
    list_tool = ListBankRecordsTool()
    add_tool = AddBankRecordTool()
    finalize_tool = FinalizeBankTool()

    # Or with a custom tool name:
    list_tool = ListBankRecordsTool(tool_name="list_items")
    ```
"""

from __future__ import annotations

import logging
import re
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


def _resolve_lookup_field(bank: Any, data: dict[str, Any]) -> str | None:
    """Determine which field to use for record lookup.

    Resolution order:
    1. Bank's ``match_fields`` (first entry)
    2. Bank schema's ``required`` fields (first entry)
    3. ``None`` if nothing is available

    Args:
        bank: The ``MemoryBank`` instance.
        data: Record data (used as last-resort fallback is not needed).

    Returns:
        The lookup field name, or ``None`` if undetermined.
    """
    if bank.match_fields:
        return bank.match_fields[0]
    required = bank.schema.get("required", [])
    if required:
        return required[0]
    return None


# Bank record IDs are 12-char hex strings (uuid4().hex[:12])
_RECORD_ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")


def _validate_record_id(record_id: str) -> dict[str, Any] | None:
    """Validate record_id format, returning an error dict if invalid.

    Bank record IDs are 12-character lowercase hex strings generated
    from ``uuid.uuid4().hex[:12]``.  This catches obviously wrong
    values (e.g. ``"1"``) with a clear message directing the model
    to use ``list_bank_records``.

    Args:
        record_id: The record_id string to validate.

    Returns:
        Error response dict if invalid, ``None`` if valid.
    """
    if not _RECORD_ID_PATTERN.match(record_id):
        return {
            "success": False,
            "error": (
                f"Invalid record_id format: '{record_id}'. "
                "Record IDs are 12-character hex strings. "
                "Use list_bank_records to see available records and their IDs."
            ),
        }
    return None


class ListBankRecordsTool(ContextAwareTool):
    """Tool for listing all records in a MemoryBank.

    Returns all records with their data fields and a total count.
    The LLM specifies which bank to list via ``bank_name``.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "list_bank_records",
            "description": "List all records in a memory bank.",
            "tags": ("wizard", "bank"),
        }

    def __init__(self, tool_name: str | None = None) -> None:
        """Initialize the tool.

        Args:
            tool_name: Custom tool name.  Defaults to ``"list_bank_records"``.
        """
        super().__init__(
            name=tool_name or "list_bank_records",
            description="List all records in a memory bank with their current values.",
        )

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "bank_name": {
                    "type": "string",
                    "description": "Name of the bank to list records from.",
                },
            },
            "required": ["bank_name"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """List all records in the specified bank.

        Args:
            context: Execution context with banks.
            **kwargs: Must include ``bank_name``.

        Returns:
            Dict with records list, count, and bank name.
        """
        bank_name = kwargs.get("bank_name")
        if not bank_name:
            return {
                "success": False,
                "error": "Missing required parameter: bank_name",
            }

        bank = _get_bank_from_context(context, bank_name)
        records = bank.all()

        items = []
        for record in records:
            item: dict[str, Any] = {"record_id": record.record_id}
            item.update(record.data)
            items.append(item)

        logger.debug(
            "Listed %d records from bank '%s'",
            len(items),
            bank_name,
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "records": items,
            "count": len(items),
            "bank_name": bank_name,
        }


class AddBankRecordTool(ContextAwareTool):
    """Tool for adding a record to a MemoryBank with duplicate detection.

    The LLM specifies which bank to add to via ``bank_name`` and passes
    field values as a ``data`` dict.  Duplicate detection uses the bank's
    ``match_fields`` configuration.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "add_bank_record",
            "description": "Add a new record to a memory bank.",
            "tags": ("wizard", "bank"),
        }

    def __init__(self, tool_name: str | None = None) -> None:
        """Initialize the tool.

        Args:
            tool_name: Custom tool name.  Defaults to ``"add_bank_record"``.
        """
        super().__init__(
            name=tool_name or "add_bank_record",
            description=(
                "Add a new record to a memory bank. "
                "Checks for duplicates by the bank's match fields."
            ),
        )

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "bank_name": {
                    "type": "string",
                    "description": "Name of the bank to add to.",
                },
                "data": {
                    "type": "object",
                    "description": "Record fields as key-value pairs.",
                },
            },
            "required": ["bank_name", "data"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add a record to the specified bank.

        Args:
            context: Execution context with banks.
            **kwargs: Must include ``bank_name`` and ``data``.

        Returns:
            Dict with add result or duplicate error.
        """
        bank_name = kwargs.get("bank_name")
        if not bank_name:
            return {
                "success": False,
                "error": "Missing required parameter: bank_name",
            }

        data = kwargs.get("data")
        if not data or not isinstance(data, dict):
            return {
                "success": False,
                "error": "Missing required parameter: data (must be a dict)",
            }

        bank = _get_bank_from_context(context, bank_name)

        # Pre-check for duplicates using the bank's lookup field
        lookup_field = _resolve_lookup_field(bank, data)
        if lookup_field:
            lookup_value = data.get(lookup_field)
            if lookup_value is not None:
                duplicates = bank.find(**{lookup_field: lookup_value})
                if duplicates:
                    existing = duplicates[0]
                    logger.debug(
                        "Duplicate detected in bank '%s': %s='%s' (record %s)",
                        bank_name,
                        lookup_field,
                        lookup_value,
                        existing.record_id,
                        extra={"conversation_id": context.conversation_id},
                    )
                    return {
                        "success": False,
                        "error": (
                            f"A record with {lookup_field}="
                            f"'{lookup_value}' already exists. "
                            f"Use the update tool to modify it."
                        ),
                        "existing_record": {
                            "record_id": existing.record_id,
                            **existing.data,
                        },
                    }

        # Pass source_stage from wizard context so tool-added records
        # carry the same provenance as collection-mode adds.
        source_stage = ""
        if context.wizard_state and context.wizard_state.current_stage:
            source_stage = context.wizard_state.current_stage

        record_id = bank.add(data, source_stage=source_stage)

        logger.debug(
            "Added record %s to bank '%s' (source_stage='%s')",
            record_id,
            bank_name,
            source_stage,
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "success": True,
            "record_id": record_id,
            "data": data,
            "total_records": bank.count(),
        }


class UpdateBankRecordTool(ContextAwareTool):
    """Tool for updating a record in a MemoryBank.

    The LLM specifies which bank via ``bank_name``, identifies the record
    by ``record_id`` (obtained from ``list_bank_records``), and passes a
    ``data`` dict with the fields to change.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "update_bank_record",
            "description": "Update an existing record in a memory bank.",
            "tags": ("wizard", "bank"),
        }

    def __init__(self, tool_name: str | None = None) -> None:
        """Initialize the tool.

        Args:
            tool_name: Custom tool name.  Defaults to ``"update_bank_record"``.
        """
        super().__init__(
            name=tool_name or "update_bank_record",
            description=(
                "Update an existing record in a memory bank. "
                "Use the record_id from list_bank_records to identify "
                "which record to update, and pass the new field values "
                "in data."
            ),
        )

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "bank_name": {
                    "type": "string",
                    "description": "Name of the bank to update in.",
                },
                "record_id": {
                    "type": "string",
                    "description": (
                        "ID of the record to update "
                        "(from list_bank_records)."
                    ),
                },
                "data": {
                    "type": "object",
                    "description": "Fields to update with new values.",
                },
            },
            "required": ["bank_name", "record_id", "data"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update a record in the specified bank.

        Args:
            context: Execution context with banks.
            **kwargs: Must include ``bank_name``, ``record_id``, and ``data``.

        Returns:
            Dict with update result or not-found error.
        """
        bank_name = kwargs.get("bank_name")
        if not bank_name:
            return {
                "success": False,
                "error": "Missing required parameter: bank_name",
            }

        record_id = kwargs.get("record_id")
        if not record_id:
            return {
                "success": False,
                "error": "Missing required parameter: record_id",
            }

        id_error = _validate_record_id(record_id)
        if id_error is not None:
            return id_error

        data = kwargs.get("data")
        if not data or not isinstance(data, dict):
            return {
                "success": False,
                "error": "Missing required parameter: data (must be a dict)",
            }

        bank = _get_bank_from_context(context, bank_name)

        record = bank.get(record_id)
        if record is None:
            return {
                "success": False,
                "error": (
                    f"No record found with record_id='{record_id}' "
                    f"in bank '{bank_name}'. Use list_bank_records "
                    "to see available records and their IDs."
                ),
            }

        updated_data = {**record.data, **data}

        # Pass modification provenance from wizard context.
        modified_in_stage = ""
        if context.wizard_state and context.wizard_state.current_stage:
            modified_in_stage = context.wizard_state.current_stage

        bank.update(record_id, updated_data, modified_in_stage=modified_in_stage)

        logger.debug(
            "Updated record %s in bank '%s' (modified_in_stage='%s')",
            record_id,
            bank_name,
            modified_in_stage,
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "success": True,
            "record_id": record_id,
            "updated_data": updated_data,
        }


class RemoveBankRecordTool(ContextAwareTool):
    """Tool for removing a record from a MemoryBank.

    The LLM specifies which bank via ``bank_name`` and identifies the
    record by ``record_id`` (obtained from ``list_bank_records``).
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "remove_bank_record",
            "description": "Remove a record from a memory bank.",
            "tags": ("wizard", "bank"),
        }

    def __init__(self, tool_name: str | None = None) -> None:
        """Initialize the tool.

        Args:
            tool_name: Custom tool name.  Defaults to ``"remove_bank_record"``.
        """
        super().__init__(
            name=tool_name or "remove_bank_record",
            description=(
                "Remove a record from a memory bank by its record_id "
                "(from list_bank_records)."
            ),
        )

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "bank_name": {
                    "type": "string",
                    "description": "Name of the bank to remove from.",
                },
                "record_id": {
                    "type": "string",
                    "description": (
                        "ID of the record to remove "
                        "(from list_bank_records)."
                    ),
                },
            },
            "required": ["bank_name", "record_id"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Remove a record from the specified bank.

        Args:
            context: Execution context with banks.
            **kwargs: Must include ``bank_name`` and ``record_id``.

        Returns:
            Dict with removal result or not-found error.
        """
        bank_name = kwargs.get("bank_name")
        if not bank_name:
            return {
                "success": False,
                "error": "Missing required parameter: bank_name",
            }

        record_id = kwargs.get("record_id")
        if not record_id:
            return {
                "success": False,
                "error": "Missing required parameter: record_id",
            }

        id_error = _validate_record_id(record_id)
        if id_error is not None:
            return id_error

        bank = _get_bank_from_context(context, bank_name)

        record = bank.get(record_id)
        if record is None:
            return {
                "success": False,
                "error": (
                    f"No record found with record_id='{record_id}' "
                    f"in bank '{bank_name}'. Use list_bank_records "
                    "to see available records and their IDs."
                ),
            }

        bank.remove(record_id)

        logger.debug(
            "Removed record %s from bank '%s'",
            record_id,
            bank_name,
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "success": True,
            "removed": {
                "record_id": record_id,
                **record.data,
            },
            "remaining_records": bank.count(),
        }


class FinalizeBankTool(ContextAwareTool):
    """Tool for confirming the bank contents are finalized.

    This is a declarative confirmation tool.  Bank data is already
    persisted in conversation metadata automatically by the wizard's
    ``_save_wizard_state``.  This tool simply returns a confirmation
    with the final record count so the LLM can present it to the user.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "finalize_bank",
            "description": "Confirm and finalize the memory bank contents.",
            "tags": ("wizard", "bank"),
        }

    def __init__(self, tool_name: str | None = None) -> None:
        """Initialize the tool.

        Args:
            tool_name: Custom tool name.  Defaults to ``"finalize_bank"``.
        """
        super().__init__(
            name=tool_name or "finalize_bank",
            description=(
                "Confirm and finalize a memory bank's contents. "
                "Data is already saved automatically."
            ),
        )

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "bank_name": {
                    "type": "string",
                    "description": "Name of the bank to finalize.",
                },
            },
            "required": ["bank_name"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Finalize the specified bank.

        Args:
            context: Execution context with banks.
            **kwargs: Must include ``bank_name``.

        Returns:
            Dict with confirmation, record count, and records.
        """
        bank_name = kwargs.get("bank_name")
        if not bank_name:
            return {
                "success": False,
                "error": "Missing required parameter: bank_name",
            }

        bank = _get_bank_from_context(context, bank_name)
        count = bank.count()
        records = bank.all()

        items = [dict(record.data) for record in records]

        logger.info(
            "Finalized bank '%s' with %d records",
            bank_name,
            count,
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "success": True,
            "finalized": True,
            "bank_name": bank_name,
            "record_count": count,
            "records": items,
        }


class CompileArtifactTool(ContextAwareTool):
    """Compile all fields and sections into the complete artifact.

    Reads the ``ArtifactBank`` from ``context.extra["artifact"]``,
    validates it, and returns the compiled output.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "compile_artifact",
            "description": "Compile the complete artifact from fields and sections.",
            "tags": ("wizard", "bank", "artifact"),
        }

    def __init__(self, tool_name: str | None = None) -> None:
        """Initialize the tool.

        Args:
            tool_name: Custom tool name.  Defaults to ``"compile_artifact"``.
        """
        super().__init__(
            name=tool_name or "compile_artifact",
            description=(
                "Compile the complete artifact from all fields and "
                "sections. Validates completeness before compiling."
            ),
        )

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
        """Compile the artifact from context.

        Args:
            context: Execution context with ``extra["artifact"]``.
            **kwargs: Not used.

        Returns:
            Dict with compiled artifact or validation errors.
        """
        artifact = context.extra.get("artifact")
        if artifact is None:
            return {
                "success": False,
                "error": (
                    "No artifact configured. "
                    "Ensure the wizard has an artifact configuration."
                ),
            }

        errors = artifact.validate()
        if errors:
            logger.debug(
                "Artifact validation failed: %s",
                errors,
                extra={"conversation_id": context.conversation_id},
            )
            return {
                "success": False,
                "errors": errors,
            }

        compiled = artifact.compile()
        logger.info(
            "Compiled artifact '%s' with %d sections",
            artifact.name,
            len(artifact.sections),
            extra={"conversation_id": context.conversation_id},
        )
        return {
            "success": True,
            "artifact": compiled,
        }
