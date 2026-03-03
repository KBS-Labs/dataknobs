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
- FinalizeArtifactTool: Validate, compile, and lock an artifact
- CompleteWizardTool: Signal wizard completion from a ReAct stage
- RestartWizardTool: Signal wizard restart from a ReAct stage

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
from collections.abc import Callable
from typing import Any

from dataknobs_llm.tools.context import ToolExecutionContext
from dataknobs_llm.tools.context_aware import ContextAwareTool

from dataknobs_bots.memory.bank import BankRecord
from dataknobs_bots.tools.response import error_response, success_response

logger = logging.getLogger(__name__)


def _get_bank_from_context(
    context: ToolExecutionContext,
    bank_name: str,
    banks_override: dict[str, Any] | None = None,
) -> Any:
    """Extract a MemoryBank from the tool execution context.

    Args:
        context: Tool execution context with ``extra["banks"]``.
        bank_name: Name of the bank to retrieve.
        banks_override: Explicit banks dict (from constructor injection).

    Returns:
        The ``MemoryBank`` instance.

    Raises:
        ValueError: If banks are not in context or the named bank is missing.
    """
    banks = banks_override or context.extra.get("banks")
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
        return error_response(
            f"Invalid record_id format: '{record_id}'. "
            "Record IDs are 12-character hex strings. "
            "Use list_bank_records to see available records and their IDs."
        )
    return None


def create_auto_save_hook(
    catalog: Any, artifact: Any
) -> Callable[[BankRecord], None]:
    """Create a lifecycle hook that auto-saves artifact to catalog.

    The hook validates the artifact and, on success, saves it to the
    catalog.  Observable: logs at INFO on save, WARNING on failure
    (not silent).

    Args:
        catalog: ``ArtifactBankCatalog`` instance.
        artifact: ``ArtifactBank`` instance.

    Returns:
        A sync hook suitable for ``MemoryBank.on_add/on_update/on_remove``.
    """

    def hook(record: BankRecord) -> None:
        errors = artifact.validate()
        if errors:
            logger.debug("Auto-save skipped (validation): %s", errors)
            return
        try:
            entry_name = catalog.save(artifact)
            logger.info("Auto-saved artifact to catalog as '%s'", entry_name)
        except Exception:
            logger.warning("Auto-save to catalog failed", exc_info=True)

    return hook


def _register_auto_save(
    bank: Any,
    context: ToolExecutionContext,
    catalog_override: Any | None = None,
    artifact_override: Any | None = None,
) -> None:
    """Register auto-save hook on bank if catalog+artifact available.

    Idempotent — uses ``key="auto_save"`` to prevent double-registration
    across multiple tool calls on the same bank instance.

    Args:
        bank: ``MemoryBank`` instance.
        context: Execution context (fallback source for catalog/artifact).
        catalog_override: Explicit catalog (from constructor injection).
        artifact_override: Explicit artifact (from constructor injection).
    """
    catalog = catalog_override or context.extra.get("catalog")
    artifact = artifact_override or context.extra.get("artifact")
    if catalog is None or artifact is None:
        return
    hook = create_auto_save_hook(catalog, artifact)
    bank.on_add(hook, key="auto_save")
    bank.on_update(hook, key="auto_save")
    bank.on_remove(hook, key="auto_save")


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
            "description": (
                "List all records in a memory bank with their current values. "
                "Call this to see record IDs before using "
                "update_bank_record or remove_bank_record."
            ),
            "tags": ("wizard", "bank"),
            "effects": ("query",),
        }

    def __init__(
        self,
        *,
        banks: dict[str, Any] | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            banks: Explicit banks dict (constructor injection).
                Falls back to ``context.extra["banks"]`` at runtime.
            tool_name: Custom tool name.  Defaults to ``"list_bank_records"``.
        """
        self._banks = banks
        super().__init__(
            name=tool_name or "list_bank_records",
            description=(
                "List all records in a memory bank with their current values. "
                "Call this to see record IDs before using "
                "update_bank_record or remove_bank_record."
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
            return error_response("Missing required parameter: bank_name")

        bank = _get_bank_from_context(context, bank_name, banks_override=self._banks)
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

        return success_response(
            records=items,
            count=len(items),
            bank_name=bank_name,
        )


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
            "description": (
                "Add a new record to a memory bank. "
                "Checks for duplicates; use update_bank_record to modify "
                "existing records instead. "
                "Auto-saves artifact to catalog on success."
            ),
            "tags": ("wizard", "bank"),
            "effects": ("mutating", "persisting"),
        }

    def __init__(
        self,
        *,
        banks: dict[str, Any] | None = None,
        catalog: Any | None = None,
        artifact: Any | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            banks: Explicit banks dict (constructor injection).
            catalog: Explicit catalog (constructor injection).
            artifact: Explicit artifact (constructor injection).
            tool_name: Custom tool name.  Defaults to ``"add_bank_record"``.
        """
        self._banks = banks
        self._catalog = catalog
        self._artifact = artifact
        super().__init__(
            name=tool_name or "add_bank_record",
            description=(
                "Add a new record to a memory bank. "
                "Checks for duplicates; use update_bank_record to modify "
                "existing records instead. "
                "Auto-saves artifact to catalog on success."
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
            return error_response("Missing required parameter: bank_name")

        data = kwargs.get("data")
        if not data or not isinstance(data, dict):
            return error_response(
                "Missing required parameter: data (must be a dict)"
            )

        bank = _get_bank_from_context(context, bank_name, banks_override=self._banks)
        _register_auto_save(
            bank, context,
            catalog_override=self._catalog,
            artifact_override=self._artifact,
        )

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
                    return error_response(
                        f"A record with {lookup_field}="
                        f"'{lookup_value}' already exists. "
                        f"Use the update tool to modify it.",
                        existing_record={
                            "record_id": existing.record_id,
                            **existing.data,
                        },
                    )

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

        return success_response(
            record_id=record_id,
            data=data,
            total_records=bank.count(),
        )


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
            "description": "Update an existing record in a memory bank. Auto-saves artifact to catalog.",
            "tags": ("wizard", "bank"),
            "effects": ("mutating", "persisting"),
        }

    def __init__(
        self,
        *,
        banks: dict[str, Any] | None = None,
        catalog: Any | None = None,
        artifact: Any | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            banks: Explicit banks dict (constructor injection).
            catalog: Explicit catalog (constructor injection).
            artifact: Explicit artifact (constructor injection).
            tool_name: Custom tool name.  Defaults to ``"update_bank_record"``.
        """
        self._banks = banks
        self._catalog = catalog
        self._artifact = artifact
        super().__init__(
            name=tool_name or "update_bank_record",
            description=(
                "Update an existing record in a memory bank. "
                "Use the record_id from list_bank_records to identify "
                "which record to update, and pass the new field values "
                "in data. Auto-saves artifact to catalog on success."
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
            return error_response("Missing required parameter: bank_name")

        record_id = kwargs.get("record_id")
        if not record_id:
            return error_response("Missing required parameter: record_id")

        id_error = _validate_record_id(record_id)
        if id_error is not None:
            return id_error

        data = kwargs.get("data")
        if not data or not isinstance(data, dict):
            return error_response(
                "Missing required parameter: data (must be a dict)"
            )

        bank = _get_bank_from_context(context, bank_name, banks_override=self._banks)
        _register_auto_save(
            bank, context,
            catalog_override=self._catalog,
            artifact_override=self._artifact,
        )

        record = bank.get(record_id)
        if record is None:
            return error_response(
                f"No record found with record_id='{record_id}' "
                f"in bank '{bank_name}'. Use list_bank_records "
                "to see available records and their IDs.",
            )

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

        return success_response(
            record_id=record_id,
            updated_data=updated_data,
        )


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
            "description": "Remove a record from a memory bank. Auto-saves artifact to catalog.",
            "tags": ("wizard", "bank"),
            "effects": ("mutating", "persisting"),
        }

    def __init__(
        self,
        *,
        banks: dict[str, Any] | None = None,
        catalog: Any | None = None,
        artifact: Any | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            banks: Explicit banks dict (constructor injection).
            catalog: Explicit catalog (constructor injection).
            artifact: Explicit artifact (constructor injection).
            tool_name: Custom tool name.  Defaults to ``"remove_bank_record"``.
        """
        self._banks = banks
        self._catalog = catalog
        self._artifact = artifact
        super().__init__(
            name=tool_name or "remove_bank_record",
            description=(
                "Remove a record from a memory bank by its record_id "
                "(from list_bank_records). "
                "Auto-saves artifact to catalog on success."
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
            return error_response("Missing required parameter: bank_name")

        record_id = kwargs.get("record_id")
        if not record_id:
            return error_response("Missing required parameter: record_id")

        id_error = _validate_record_id(record_id)
        if id_error is not None:
            return id_error

        bank = _get_bank_from_context(context, bank_name, banks_override=self._banks)
        _register_auto_save(
            bank, context,
            catalog_override=self._catalog,
            artifact_override=self._artifact,
        )

        record = bank.get(record_id)
        if record is None:
            return error_response(
                f"No record found with record_id='{record_id}' "
                f"in bank '{bank_name}'. Use list_bank_records "
                "to see available records and their IDs.",
            )

        bank.remove(record_id)

        logger.debug(
            "Removed record %s from bank '%s'",
            record_id,
            bank_name,
            extra={"conversation_id": context.conversation_id},
        )

        return success_response(
            removed={
                "record_id": record_id,
                **record.data,
            },
            remaining_records=bank.count(),
        )


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
            "effects": ("locking",),
        }

    def __init__(
        self,
        *,
        banks: dict[str, Any] | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            banks: Explicit banks dict (constructor injection).
            tool_name: Custom tool name.  Defaults to ``"finalize_bank"``.
        """
        self._banks = banks
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
            return error_response("Missing required parameter: bank_name")

        bank = _get_bank_from_context(context, bank_name, banks_override=self._banks)
        count = bank.count()
        records = bank.all()

        items = [dict(record.data) for record in records]

        logger.info(
            "Finalized bank '%s' with %d records",
            bank_name,
            count,
            extra={"conversation_id": context.conversation_id},
        )

        return success_response(
            finalized=True,
            bank_name=bank_name,
            record_count=count,
            records=items,
        )


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
            "description": (
                "Compile the complete artifact from all fields and sections. "
                "Validates completeness before compiling. "
                "Use finalize_artifact to lock, then save_to_catalog to persist."
            ),
            "tags": ("wizard", "bank", "artifact"),
            "effects": ("query",),
        }

    def __init__(
        self,
        *,
        artifact: Any | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            artifact: Explicit artifact (constructor injection).
            tool_name: Custom tool name.  Defaults to ``"compile_artifact"``.
        """
        self._artifact = artifact
        super().__init__(
            name=tool_name or "compile_artifact",
            description=(
                "Compile the complete artifact from all fields and "
                "sections. Validates completeness before compiling. "
                "Use finalize_artifact to lock, then "
                "save_to_catalog to persist."
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
        artifact = self._artifact or context.extra.get("artifact")
        if artifact is None:
            return error_response(
                "No artifact configured. "
                "Ensure the wizard has an artifact configuration.",
            )

        errors = artifact.validate()
        if errors:
            logger.debug(
                "Artifact validation failed: %s",
                errors,
                extra={"conversation_id": context.conversation_id},
            )
            return error_response(
                "Artifact validation failed",
                errors=errors,
            )

        compiled = artifact.compile()
        logger.info(
            "Compiled artifact '%s' with %d sections",
            artifact.name,
            len(artifact.sections),
            extra={"conversation_id": context.conversation_id},
        )
        return success_response(artifact=compiled)


class FinalizeArtifactTool(ContextAwareTool):
    """Validate, compile, and lock the artifact against further edits.

    Reads the ``ArtifactBank`` from ``context.extra["artifact"]``,
    calls ``artifact.finalize()`` which validates, compiles, and sets
    ``_finalized = True``.

    Distinct from ``CompileArtifactTool``: compile = preview (no lock),
    finalize = commit (locks the artifact).
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "finalize_artifact",
            "description": (
                "Validate, compile, and lock the artifact. "
                "No further edits allowed after finalization. "
                "Use save_to_catalog to persist the finalized artifact."
            ),
            "tags": ("wizard", "bank", "artifact"),
            "effects": ("locking",),
        }

    def __init__(
        self,
        *,
        artifact: Any | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            artifact: Explicit artifact (constructor injection).
            tool_name: Custom tool name.  Defaults to ``"finalize_artifact"``.
        """
        self._artifact = artifact
        super().__init__(
            name=tool_name or "finalize_artifact",
            description=(
                "Validate, compile, and lock the artifact. "
                "No further edits allowed after finalization. "
                "Use save_to_catalog to persist the finalized artifact."
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
        """Finalize the artifact from context.

        Args:
            context: Execution context with ``extra["artifact"]``.
            **kwargs: Not used.

        Returns:
            Dict with compiled artifact on success, or errors on failure.
        """
        artifact = self._artifact or context.extra.get("artifact")
        if artifact is None:
            return error_response(
                "No artifact configured. "
                "Ensure the wizard has an artifact configuration.",
            )

        # Idempotent: already finalized -> return compiled without error
        if artifact.is_finalized:
            compiled = artifact.compile()
            logger.debug(
                "Artifact '%s' already finalized, returning compiled",
                artifact.name,
                extra={"conversation_id": context.conversation_id},
            )
            return success_response(
                already_finalized=True,
                artifact=compiled,
            )

        # Validate before finalizing — return errors without locking
        errors = artifact.validate()
        if errors:
            logger.debug(
                "Artifact finalization failed validation: %s",
                errors,
                extra={"conversation_id": context.conversation_id},
            )
            return error_response(
                "Artifact validation failed",
                errors=errors,
            )

        compiled = artifact.finalize()
        logger.info(
            "Finalized artifact '%s' with %d sections",
            artifact.name,
            len(artifact.sections),
            extra={"conversation_id": context.conversation_id},
        )
        return success_response(
            is_finalized=True,
            artifact=compiled,
        )


class CompleteWizardTool(ContextAwareTool):
    """Signal wizard completion from within a ReAct stage.

    Sets a completion signal in ``context.extra["_completion_signal"]``
    that the wizard checks after the ReAct loop returns.  If an
    unfinalised artifact exists, it is auto-finalized for convenience.

    Works with or without an artifact — wizard completion is a lifecycle
    concern independent of artifact finalization.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "complete_wizard",
            "description": (
                "Signal that the wizard workflow is complete. "
                "Auto-finalizes the artifact if not already finalized. "
                "Call save_to_catalog first to persist the artifact."
            ),
            "tags": ("wizard",),
            "effects": ("signaling",),
        }

    def __init__(
        self,
        *,
        artifact: Any | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            artifact: Explicit artifact (constructor injection).
            tool_name: Custom tool name.  Defaults to ``"complete_wizard"``.
        """
        self._artifact = artifact
        super().__init__(
            name=tool_name or "complete_wizard",
            description=(
                "Signal that the wizard workflow is complete. "
                "Auto-finalizes the artifact if not already finalized. "
                "Call save_to_catalog first to persist the artifact."
            ),
        )

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "Optional closing notes or summary of the "
                        "completed workflow."
                    ),
                },
            },
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Signal wizard completion.

        Args:
            context: Execution context with ``extra["_completion_signal"]``.
            **kwargs: Optional ``summary``.

        Returns:
            Dict confirming completion signal was set.
        """
        signal = context.extra.get("_completion_signal")
        if signal is None:
            return error_response(
                "No completion signal available. "
                "This tool can only be used within a wizard ReAct stage.",
            )

        # Idempotent: already requested
        if signal.get("requested"):
            return success_response(already_completed=True)

        summary = kwargs.get("summary", "")

        # Auto-finalize artifact if present and not yet finalized
        artifact = self._artifact or context.extra.get("artifact")
        if artifact is not None and not artifact.is_finalized:
            errors = artifact.validate()
            if errors:
                logger.debug(
                    "Cannot complete wizard — artifact validation failed: %s",
                    errors,
                    extra={"conversation_id": context.conversation_id},
                )
                return error_response(
                    "Cannot complete wizard: artifact validation failed.",
                    artifact_errors=errors,
                )
            artifact.finalize()
            logger.info(
                "Auto-finalized artifact '%s' during wizard completion",
                artifact.name,
                extra={"conversation_id": context.conversation_id},
            )

        # Set the signal for the wizard to pick up
        signal["requested"] = True
        signal["summary"] = summary

        logger.info(
            "Wizard completion signaled via complete_wizard tool",
            extra={"conversation_id": context.conversation_id},
        )
        return success_response(completed=True, summary=summary)


class RestartWizardTool(ContextAwareTool):
    """Signal wizard restart from within a ReAct stage.

    Sets a restart signal in ``context.extra["_restart_signal"]``
    that the wizard checks after the ReAct loop returns.  The wizard
    then delegates to its existing ``_execute_restart()`` which clears
    all banks, resets the artifact, and returns the FSM to the start
    stage.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "restart_wizard",
            "description": (
                "Restart the wizard from the beginning. "
                "Clears all data and returns to the first stage."
            ),
            "tags": ("wizard",),
            "effects": ("signaling",),
        }

    def __init__(self, tool_name: str | None = None) -> None:
        """Initialize the tool.

        Args:
            tool_name: Custom tool name.  Defaults to ``"restart_wizard"``.
        """
        super().__init__(
            name=tool_name or "restart_wizard",
            description=(
                "Restart the wizard from the beginning, "
                "clearing all collected data."
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
        """Signal wizard restart.

        Args:
            context: Execution context with ``extra["_restart_signal"]``.
            **kwargs: Not used.

        Returns:
            Dict confirming restart signal was set.
        """
        signal = context.extra.get("_restart_signal")
        if signal is None:
            return error_response(
                "No restart signal available. "
                "This tool can only be used within a wizard ReAct stage.",
            )

        # Idempotent: already requested
        if signal.get("requested"):
            return success_response(already_requested=True)

        signal["requested"] = True

        logger.info(
            "Wizard restart signaled via restart_wizard tool",
            extra={"conversation_id": context.conversation_id},
        )
        return success_response(restarting=True)
