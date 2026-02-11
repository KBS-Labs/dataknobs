"""Generic knowledge base resource tools for wizard-driven bot configuration.

Provides reusable LLM-callable tools for managing RAG resources during
wizard flows. These tools operate on wizard collected data to track
knowledge sources, add/remove resources, and trigger ingestion.

All tools use ``_get_wizard_data_ref()`` from ``config_tools`` to mutate
wizard state directly (not a copy), since they need to persist changes
across tool invocations within a wizard session.

Tools:
- CheckKnowledgeSourceTool: Verify a knowledge source directory exists
- ListKBResourcesTool: List currently tracked KB resources
- AddKBResourceTool: Add a resource to the KB resource list
- RemoveKBResourceTool: Remove a resource from the KB resource list
- IngestKnowledgeBaseTool: Write ingestion manifest and finalize KB config

Example:
    ```python
    from pathlib import Path
    from dataknobs_bots.tools.kb_tools import (
        CheckKnowledgeSourceTool, AddKBResourceTool, IngestKnowledgeBaseTool,
    )

    check_tool = CheckKnowledgeSourceTool()
    add_tool = AddKBResourceTool(knowledge_dir=Path("/data/knowledge"))
    ingest_tool = IngestKnowledgeBaseTool(knowledge_dir=Path("/data/knowledge"))
    ```
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dataknobs_llm.tools.context import ToolExecutionContext
from dataknobs_llm.tools.context_aware import ContextAwareTool

from .config_tools import _get_wizard_data_ref

logger = logging.getLogger(__name__)

# Default file glob patterns for knowledge sources
_DEFAULT_GLOB_PATTERNS = ["*.md", "*.txt", "*.pdf", "*.html", "*.json", "*.csv"]


def _resolve_knowledge_dir(
    constructor_dir: Path | None,
    wizard_data: dict[str, Any],
) -> Path | None:
    """Resolve the knowledge directory from constructor param or wizard data.

    Args:
        constructor_dir: Directory passed to tool constructor (takes priority).
        wizard_data: Wizard collected data (fallback: ``_knowledge_dir`` key).

    Returns:
        Resolved Path, or None if neither source provides a value.
    """
    if constructor_dir is not None:
        return constructor_dir
    wd_dir = wizard_data.get("_knowledge_dir")
    if wd_dir is not None:
        return Path(wd_dir)
    return None


class CheckKnowledgeSourceTool(ContextAwareTool):
    """Tool for verifying a knowledge source directory exists and has content.

    Checks the specified path for files matching common document patterns
    and records the results in wizard data for subsequent tools.

    Wizard data written:
    - ``source_verified``: bool — whether the source was found
    - ``files_found``: list[str] — matching file names
    - ``_source_path_resolved``: str — the resolved absolute path
    - ``_kb_resources``: list[dict] — initialized if not present
    """

    def __init__(self) -> None:
        """Initialize the tool."""
        super().__init__(
            name="check_knowledge_source",
            description=(
                "Check if a knowledge source directory exists and contains "
                "files that can be used for the knowledge base."
            ),
        )

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "Path to the knowledge source directory",
                },
                "file_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Glob patterns for files to look for "
                        "(default: *.md, *.txt, *.pdf, *.html, *.json, *.csv)"
                    ),
                },
            },
            "required": ["source_path"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        source_path: str,
        file_patterns: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Check the knowledge source directory.

        Args:
            context: Execution context with wizard state.
            source_path: Path to the knowledge source directory.
            file_patterns: Optional glob patterns to match files.

        Returns:
            Dict with verification results.
        """
        wizard_data = _get_wizard_data_ref(context)
        patterns = file_patterns or _DEFAULT_GLOB_PATTERNS

        path = Path(source_path).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            wizard_data["source_verified"] = False
            wizard_data["files_found"] = []
            logger.debug(
                "Knowledge source not found: %s",
                path,
                extra={"conversation_id": context.conversation_id},
            )
            return {
                "exists": False,
                "error": f"Directory not found: {source_path}",
                "files_found": [],
            }

        # Find matching files
        found_files: list[str] = []
        for pattern in patterns:
            for match in path.glob(pattern):
                if match.is_file():
                    found_files.append(match.name)
        found_files = sorted(set(found_files))

        # Update wizard data
        wizard_data["source_verified"] = True
        wizard_data["files_found"] = found_files
        wizard_data["_source_path_resolved"] = str(path)
        if "_kb_resources" not in wizard_data:
            wizard_data["_kb_resources"] = []

        logger.debug(
            "Checked knowledge source: %s (%d files)",
            path,
            len(found_files),
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "exists": True,
            "path": str(path),
            "files_found": found_files,
            "file_count": len(found_files),
            "patterns_checked": patterns,
        }


class ListKBResourcesTool(ContextAwareTool):
    """Tool for listing currently tracked knowledge base resources.

    Reads ``_kb_resources`` and ``_source_path_resolved`` from wizard data
    to show what resources have been added so far.
    """

    def __init__(self) -> None:
        """Initialize the tool."""
        super().__init__(
            name="list_kb_resources",
            description=(
                "List the knowledge base resources that have been added "
                "to the current bot configuration."
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
        """List KB resources.

        Args:
            context: Execution context with wizard state.

        Returns:
            Dict with resource list and source path.
        """
        wizard_data = _get_wizard_data_ref(context)
        resources = wizard_data.get("_kb_resources", [])
        source_path = wizard_data.get("_source_path_resolved")

        logger.debug(
            "Listed %d KB resources",
            len(resources),
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "resources": resources,
            "count": len(resources),
            "source_path": source_path,
        }


class AddKBResourceTool(ContextAwareTool):
    """Tool for adding a resource to the knowledge base resource list.

    Supports adding file references (from the source directory) or inline
    content that gets written to the knowledge directory.

    Wizard data read/written:
    - ``_kb_resources``: list[dict] — resource list (append)
    - ``domain_id``: str — used for knowledge directory organization

    Attributes:
        _knowledge_dir: Optional base directory for writing inline content.
    """

    def __init__(self, knowledge_dir: Path | None = None) -> None:
        """Initialize the tool.

        Args:
            knowledge_dir: Base directory for knowledge files. Used when
                writing inline content to disk. Resolved from wizard data
                ``_knowledge_dir`` if not provided here.
        """
        super().__init__(
            name="add_kb_resource",
            description=(
                "Add a resource to the bot's knowledge base. Can add "
                "a file from the source directory or inline content."
            ),
        )
        self._knowledge_dir = knowledge_dir

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name/filename of the resource",
                },
                "resource_type": {
                    "type": "string",
                    "description": "Type of resource: 'file' or 'inline'",
                    "enum": ["file", "inline"],
                    "default": "file",
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Inline content to write (only for resource_type='inline')"
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Optional description of the resource",
                },
            },
            "required": ["name"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        name: str,
        resource_type: str = "file",
        content: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add a KB resource.

        Args:
            context: Execution context with wizard state.
            name: Resource name/filename.
            resource_type: Type of resource ('file' or 'inline').
            content: Inline content (required if resource_type='inline').
            description: Optional resource description.

        Returns:
            Dict with add result.
        """
        wizard_data = _get_wizard_data_ref(context)
        resources: list[dict[str, Any]] = wizard_data.setdefault(
            "_kb_resources", []
        )

        # Check for duplicate
        existing_names = {r["name"] for r in resources}
        if name in existing_names:
            return {
                "success": False,
                "error": f"Resource already exists: {name}",
                "existing_resources": len(resources),
            }

        resource: dict[str, Any] = {
            "name": name,
            "type": resource_type,
        }
        if description:
            resource["description"] = description

        # Handle inline content — write to knowledge directory
        if resource_type == "inline":
            if not content:
                return {
                    "success": False,
                    "error": "Content is required for inline resources",
                }
            kb_dir = _resolve_knowledge_dir(self._knowledge_dir, wizard_data)
            if kb_dir is None:
                return {
                    "success": False,
                    "error": "No knowledge directory configured",
                }
            domain_id = wizard_data.get("domain_id", "default")
            target_dir = kb_dir / domain_id
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / name
            target_path.write_text(content, encoding="utf-8")
            resource["path"] = str(target_path)
            logger.debug(
                "Wrote inline resource: %s",
                target_path,
                extra={"conversation_id": context.conversation_id},
            )

        resources.append(resource)

        logger.debug(
            "Added KB resource: %s (type=%s)",
            name,
            resource_type,
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "success": True,
            "resource": resource,
            "total_resources": len(resources),
        }


class RemoveKBResourceTool(ContextAwareTool):
    """Tool for removing a resource from the knowledge base resource list.

    Wizard data read/written:
    - ``_kb_resources``: list[dict] — resource list (remove by name)
    """

    def __init__(self) -> None:
        """Initialize the tool."""
        super().__init__(
            name="remove_kb_resource",
            description=(
                "Remove a resource from the bot's knowledge base "
                "resource list."
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
                    "description": "Name of the resource to remove",
                },
            },
            "required": ["name"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Remove a KB resource.

        Args:
            context: Execution context with wizard state.
            name: Name of the resource to remove.

        Returns:
            Dict with removal result.
        """
        wizard_data = _get_wizard_data_ref(context)
        resources: list[dict[str, Any]] = wizard_data.get("_kb_resources", [])

        original_count = len(resources)
        updated = [r for r in resources if r["name"] != name]

        if len(updated) == original_count:
            return {
                "success": False,
                "error": f"Resource not found: {name}",
                "available": [r["name"] for r in resources],
            }

        wizard_data["_kb_resources"] = updated

        logger.debug(
            "Removed KB resource: %s",
            name,
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "success": True,
            "removed": name,
            "remaining_resources": len(updated),
        }


class IngestKnowledgeBaseTool(ContextAwareTool):
    """Tool for writing the KB ingestion manifest and finalizing KB config.

    Writes a ``manifest.json`` file listing resources and chunking
    parameters, and updates wizard data with the final KB configuration
    for inclusion in the bot config.

    Wizard data read:
    - ``_kb_resources``: list[dict] — resources to include
    - ``domain_id``: str — domain identifier
    - ``files_found``: list[str] — auto-discovered files (fallback)
    - ``_source_path_resolved``: str — resolved source path

    Wizard data written:
    - ``kb_config``: dict — final KB configuration for the bot config
    - ``kb_resources``: list[dict] — finalized resource list (public key)
    - ``ingestion_complete``: bool — whether ingestion manifest was written

    Attributes:
        _knowledge_dir: Optional base directory for knowledge files.
    """

    def __init__(self, knowledge_dir: Path | None = None) -> None:
        """Initialize the tool.

        Args:
            knowledge_dir: Base directory for knowledge files. Resolved
                from wizard data ``_knowledge_dir`` if not provided here.
        """
        super().__init__(
            name="ingest_knowledge_base",
            description=(
                "Finalize and ingest the knowledge base resources. "
                "Writes an ingestion manifest and prepares the KB "
                "configuration for the bot."
            ),
        )
        self._knowledge_dir = knowledge_dir

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "chunk_size": {
                    "type": "integer",
                    "description": "Size of text chunks for ingestion",
                    "default": 512,
                },
                "chunk_overlap": {
                    "type": "integer",
                    "description": "Overlap between consecutive chunks",
                    "default": 50,
                },
            },
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Write ingestion manifest and finalize KB config.

        Args:
            context: Execution context with wizard state.
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.

        Returns:
            Dict with ingestion result.
        """
        wizard_data = _get_wizard_data_ref(context)
        domain_id = wizard_data.get("domain_id", "default")
        resources = wizard_data.get("_kb_resources", [])
        source_path = wizard_data.get("_source_path_resolved")

        # Fallback: if no explicit resources, use auto-discovered files
        if not resources and wizard_data.get("files_found"):
            resources = [
                {"name": f, "type": "file"}
                for f in wizard_data["files_found"]
            ]

        if not resources:
            return {
                "success": False,
                "error": "No resources to ingest. Add resources first.",
            }

        kb_dir = _resolve_knowledge_dir(self._knowledge_dir, wizard_data)
        if kb_dir is None:
            return {
                "success": False,
                "error": "No knowledge directory configured",
            }

        # Write manifest
        manifest_dir = kb_dir / domain_id
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "domain_id": domain_id,
            "source_path": source_path,
            "resources": resources,
            "chunking": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
        }
        manifest_path = manifest_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        # Build KB config for the bot configuration
        kb_config: dict[str, Any] = {
            "enabled": True,
            "type": "rag",
            "chunking": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
        }

        # Update wizard data with finalized KB config
        wizard_data["kb_config"] = kb_config
        wizard_data["kb_resources"] = resources
        wizard_data["ingestion_complete"] = True

        logger.info(
            "Wrote KB manifest for '%s' with %d resources",
            domain_id,
            len(resources),
            extra={
                "domain_id": domain_id,
                "resource_count": len(resources),
                "conversation_id": context.conversation_id,
            },
        )

        return {
            "success": True,
            "domain_id": domain_id,
            "manifest_path": str(manifest_path),
            "resource_count": len(resources),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
