"""Catalog tools — LLM-callable tools for ArtifactBankCatalog operations.

Provides tools for listing, saving to, and loading from an
``ArtifactBankCatalog`` that is injected into the tool execution context
via ``context.extra["catalog"]``.

Tools:
- ListCatalogTool: List all entries in the catalog
- SaveToCatalogTool: Validate and save current artifact to catalog
- LoadFromCatalogTool: Load a catalog entry into the current artifact

Example::

    from dataknobs_bots.tools.catalog_tools import (
        ListCatalogTool, SaveToCatalogTool, LoadFromCatalogTool,
    )

    list_tool = ListCatalogTool()
    save_tool = SaveToCatalogTool()
    load_tool = LoadFromCatalogTool()
"""

from __future__ import annotations

import logging
from typing import Any

from dataknobs_llm.tools.context import ToolExecutionContext
from dataknobs_llm.tools.context_aware import ContextAwareTool

from dataknobs_bots.tools.response import error_response, success_response

logger = logging.getLogger(__name__)


def _get_catalog_from_context(
    context: ToolExecutionContext,
    catalog_override: Any | None = None,
) -> Any:
    """Extract ArtifactBankCatalog from tool execution context.

    Args:
        context: Tool execution context with ``extra["catalog"]``.
        catalog_override: Explicit catalog (from constructor injection).

    Returns:
        The ``ArtifactBankCatalog`` instance.

    Raises:
        ValueError: If catalog is not in context.
    """
    catalog = catalog_override or context.extra.get("catalog")
    if catalog is None:
        raise ValueError(
            "Catalog tools require a catalog in execution context. "
            "Ensure the wizard has catalog configured in artifact settings."
        )
    return catalog


def _get_artifact_from_context(
    context: ToolExecutionContext,
    artifact_override: Any | None = None,
) -> Any:
    """Extract ArtifactBank from tool execution context.

    Args:
        context: Tool execution context with ``extra["artifact"]``.
        artifact_override: Explicit artifact (from constructor injection).

    Returns:
        The ``ArtifactBank`` instance.

    Raises:
        ValueError: If artifact is not in context.
    """
    artifact = artifact_override or context.extra.get("artifact")
    if artifact is None:
        raise ValueError(
            "Catalog tools require an artifact in execution context. "
            "Ensure the wizard has an artifact configuration."
        )
    return artifact


class ListCatalogTool(ContextAwareTool):
    """List all entries in the artifact catalog.

    Reads the catalog from ``context.extra["catalog"]`` and returns
    summary information for each stored artifact.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "list_catalog",
            "description": "List all saved artifacts in the catalog.",
            "tags": ("wizard", "catalog"),
        }

    def __init__(
        self,
        *,
        catalog: Any | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            catalog: Explicit catalog (constructor injection).
            tool_name: Custom tool name.  Defaults to ``"list_catalog"``.
        """
        self._catalog = catalog
        super().__init__(
            name=tool_name or "list_catalog",
            description=(
                "List all saved artifacts in the catalog with their "
                "names, sections, and field counts."
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
        """List all catalog entries.

        Args:
            context: Execution context with catalog.
            **kwargs: Not used.

        Returns:
            Dict with entries list and count.
        """
        try:
            catalog = _get_catalog_from_context(
                context, catalog_override=self._catalog,
            )
        except ValueError as exc:
            return error_response(str(exc))

        entries = catalog.list()
        logger.debug(
            "Listed %d catalog entries",
            len(entries),
            extra={"conversation_id": context.conversation_id},
        )
        return {
            "entries": entries,
            "count": len(entries),
        }


class SaveToCatalogTool(ContextAwareTool):
    """Save the current artifact to the catalog.

    Reads the artifact from ``context.extra["artifact"]`` and the
    catalog from ``context.extra["catalog"]``, validates the artifact,
    compiles it, and saves it to the catalog.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "save_to_catalog",
            "description": "Save the current artifact to the catalog.",
            "tags": ("wizard", "catalog"),
        }

    def __init__(
        self,
        *,
        catalog: Any | None = None,
        artifact: Any | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            catalog: Explicit catalog (constructor injection).
            artifact: Explicit artifact (constructor injection).
            tool_name: Custom tool name.  Defaults to ``"save_to_catalog"``.
        """
        self._catalog = catalog
        self._artifact = artifact
        super().__init__(
            name=tool_name or "save_to_catalog",
            description=(
                "Validate and save the current artifact to the catalog. "
                "Auto-finalizes artifact if not already finalized."
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
        """Save artifact to catalog.

        Args:
            context: Execution context with artifact and catalog.
            **kwargs: Not used.

        Returns:
            Dict with success status and artifact info, or errors.
        """
        try:
            artifact = _get_artifact_from_context(
                context, artifact_override=self._artifact,
            )
        except ValueError as exc:
            return error_response(str(exc))

        try:
            catalog = _get_catalog_from_context(
                context, catalog_override=self._catalog,
            )
        except ValueError as exc:
            return error_response(str(exc))

        errors = artifact.validate()
        if errors:
            logger.debug(
                "Artifact validation failed for catalog save: %s",
                errors,
                extra={"conversation_id": context.conversation_id},
            )
            return error_response(
                "Artifact validation failed",
                errors=errors,
            )

        entry_name = catalog.save(artifact)

        # Auto-finalize after explicit save — triggers the auto-restart
        # guard in wizard.py when the next user message arrives, so the
        # LLM doesn't need to remember to call finalize_artifact +
        # complete_wizard separately.
        if not artifact.is_finalized:
            try:
                artifact.finalize()
                logger.info(
                    "Auto-finalized artifact after catalog save",
                    extra={"conversation_id": context.conversation_id},
                )
            except ValueError:
                pass  # Validation errors — already saved, just can't finalize

        logger.info(
            "Saved artifact '%s' to catalog as '%s'",
            artifact.name,
            entry_name,
            extra={"conversation_id": context.conversation_id},
        )
        return success_response(
            name=entry_name,
            catalog_count=catalog.count(),
        )


class LoadFromCatalogTool(ContextAwareTool):
    """Load a catalog entry into the current artifact.

    Reads the catalog and artifact from the execution context, finds the
    named entry, and replaces the artifact's state with the loaded data
    via ``catalog.load_into()``.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "load_from_catalog",
            "description": "Load a saved artifact from the catalog.",
            "tags": ("wizard", "catalog"),
        }

    def __init__(
        self,
        *,
        catalog: Any | None = None,
        artifact: Any | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            catalog: Explicit catalog (constructor injection).
            artifact: Explicit artifact (constructor injection).
            tool_name: Custom tool name.  Defaults to ``"load_from_catalog"``.
        """
        self._catalog = catalog
        self._artifact = artifact
        super().__init__(
            name=tool_name or "load_from_catalog",
            description=(
                "Load a previously saved artifact from the catalog by name. "
                "Replaces the current artifact's fields and sections."
            ),
        )

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the catalog entry to load.",
                },
            },
            "required": ["name"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Load a catalog entry into the current artifact.

        Args:
            context: Execution context with catalog and artifact.
            **kwargs: Must include ``name``.

        Returns:
            Dict with loaded artifact summary, or not-found error.
        """
        entry_name = kwargs.get("name")
        if not entry_name:
            return error_response("Missing required parameter: name")

        try:
            catalog = _get_catalog_from_context(
                context, catalog_override=self._catalog,
            )
        except ValueError as exc:
            return error_response(str(exc))

        try:
            artifact = _get_artifact_from_context(
                context, artifact_override=self._artifact,
            )
        except ValueError as exc:
            return error_response(str(exc))

        found = catalog.load_into(entry_name, artifact)
        if not found:
            available = [e["name"] for e in catalog.list()]
            return error_response(
                f"No artifact named '{entry_name}' in catalog. "
                f"Available: {available}",
            )

        compiled = artifact.compile()
        section_summary: dict[str, int] = {}
        for key, value in compiled.items():
            if key.startswith("_"):
                continue
            if isinstance(value, list):
                section_summary[key] = len(value)

        logger.info(
            "Loaded artifact '%s' from catalog",
            entry_name,
            extra={"conversation_id": context.conversation_id},
        )
        return success_response(
            loaded={
                "name": entry_name,
                "fields": {
                    k: v for k, v in compiled.items()
                    if not k.startswith("_")
                    and not isinstance(v, list)
                },
                "sections": section_summary,
            },
        )
