"""Tool catalog mapping tool names to class paths and default configuration.

Provides a single source of truth for tool metadata, enabling config
builders to reference tools by name and produce correct bot/wizard configs.

Built on ``Registry[ToolEntry]`` for thread safety, metrics, and consistent
error handling.

Example:
    ```python
    from dataknobs_bots.config import default_catalog, create_default_catalog

    # Look up a tool's config
    config = default_catalog.to_bot_config("knowledge_search", k=10)
    # {"class": "dataknobs_bots.tools.knowledge_search.KnowledgeSearchTool",
    #  "params": {"k": 10}}

    # Extend with custom tools
    catalog = create_default_catalog()
    catalog.register_tool(
        name="calculator",
        class_path="myapp.tools.CalculatorTool",
        description="Perform math calculations.",
        tags=("educational",),
    )
    ```
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from dataknobs_common.registry import Registry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolEntry:
    """Metadata for a tool in the catalog.

    Captures the information needed to:
    - Generate bot config entries (class path + params)
    - Reference tools in wizard stage configs (name)
    - Discover tools by capability (tags)
    - Validate tool dependencies (requires)
    """

    name: str
    class_path: str
    description: str = ""
    default_params: dict[str, Any] = None  # type: ignore[assignment]
    tags: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Set default_params to empty dict if None."""
        if self.default_params is None:
            object.__setattr__(self, "default_params", {})

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (suitable for YAML output).

        Omits empty/default fields for clean output.
        """
        result: dict[str, Any] = {
            "name": self.name,
            "class_path": self.class_path,
        }
        if self.description:
            result["description"] = self.description
        if self.default_params:
            result["default_params"] = dict(self.default_params)
        if self.tags:
            result["tags"] = sorted(self.tags)
        if self.requires:
            result["requires"] = sorted(self.requires)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolEntry:
        """Deserialize from dict (e.g., loaded from YAML)."""
        return cls(
            name=data["name"],
            class_path=data["class_path"],
            description=data.get("description", ""),
            default_params=data.get("default_params") or {},
            tags=frozenset(data.get("tags") or ()),
            requires=frozenset(data.get("requires") or ()),
        )

    def to_bot_config(self, **param_overrides: Any) -> dict[str, Any]:
        """Generate a bot config tool entry.

        Returns a dict suitable for ``DynaBot._resolve_tool()``:
        ``{"class": "full.class.path", "params": {...}}``

        Args:
            **param_overrides: Override default params.

        Returns:
            Bot config dict for this tool.
        """
        params = dict(self.default_params)
        params.update(param_overrides)
        config: dict[str, Any] = {"class": self.class_path}
        if params:
            config["params"] = params
        return config


@runtime_checkable
class CatalogDescribable(Protocol):
    """Protocol for tool classes that declare their own catalog metadata.

    Tool classes implementing this protocol can be registered via
    ``ToolCatalog.register_from_class()``, which auto-computes the
    class path from the class's module and qualname.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class.

        Expected keys:
            name: str           -- Tool's runtime name
            description: str    -- Human-readable description
            default_params: dict -- Default constructor params (optional)
            tags: Sequence[str] -- Categorization tags (optional)
            requires: Sequence[str] -- Dependencies (optional)

        Note:
            ``class_path`` is computed automatically from the class's
            module and qualname -- tools should NOT include it.
        """
        ...


class ToolCatalog(Registry[ToolEntry]):
    """Registry mapping tool names to class paths and default configuration.

    Provides a single source of truth for tool metadata, enabling config
    builders to reference tools by name and produce correct bot/wizard configs.

    Built on ``Registry[ToolEntry]`` for thread safety, metrics, and
    consistent error handling.

    Example:
        ```python
        catalog = ToolCatalog()
        catalog.register_tool(
            name="knowledge_search",
            class_path="dataknobs_bots.tools.knowledge_search.KnowledgeSearchTool",
            description="Search the knowledge base.",
            tags=("general", "rag"),
            requires=("knowledge_base",),
        )
        config = catalog.to_bot_config("knowledge_search", k=10)
        ```
    """

    def __init__(self) -> None:
        """Initialize the catalog."""
        super().__init__("tool_catalog", enable_metrics=True)

    # -- Registration (convenience over base register()) --

    def register_tool(
        self,
        name: str,
        class_path: str,
        description: str = "",
        default_params: dict[str, Any] | None = None,
        tags: Sequence[str] = (),
        requires: Sequence[str] = (),
    ) -> None:
        """Register a tool in the catalog.

        Args:
            name: Tool's runtime name.
            class_path: Fully-qualified class path.
            description: Human-readable description.
            default_params: Default constructor params.
            tags: Categorization tags.
            requires: Dependency identifiers.
        """
        entry = ToolEntry(
            name=name,
            class_path=class_path,
            description=description,
            default_params=default_params or {},
            tags=frozenset(tags),
            requires=frozenset(requires),
        )
        self.register(name, entry)

    def register_entry(self, entry: ToolEntry) -> None:
        """Register a pre-built ToolEntry.

        Args:
            entry: ToolEntry to register.
        """
        self.register(entry.name, entry)

    def register_from_dict(self, data: dict[str, Any]) -> None:
        """Register a tool from a dict (e.g., loaded from YAML).

        Args:
            data: Dict with ``name`` and ``class_path`` keys.
        """
        entry = ToolEntry.from_dict(data)
        self.register(entry.name, entry)

    def register_many_from_dicts(self, entries: list[dict[str, Any]]) -> None:
        """Register multiple tools from dicts.

        Args:
            entries: List of tool definition dicts.
        """
        for data in entries:
            self.register_from_dict(data)

    def register_from_class(self, tool_class: type) -> None:
        """Register a tool class that provides ``catalog_metadata()``.

        Computes ``class_path`` automatically from the class's module path.

        Args:
            tool_class: A tool class with a ``catalog_metadata()`` classmethod.

        Raises:
            ValueError: If tool_class does not implement ``catalog_metadata()``.
        """
        if not hasattr(tool_class, "catalog_metadata") or not callable(
            tool_class.catalog_metadata
        ):
            raise ValueError(
                f"{tool_class.__name__} does not implement catalog_metadata()"
            )
        meta = tool_class.catalog_metadata()
        class_path = f"{tool_class.__module__}.{tool_class.__qualname__}"
        self.register_tool(
            name=meta["name"],
            class_path=class_path,
            description=meta.get("description", ""),
            default_params=meta.get("default_params"),
            tags=meta.get("tags", ()),
            requires=meta.get("requires", ()),
        )

    # -- Query --

    def list_tools(
        self,
        tags: Sequence[str] | None = None,
    ) -> list[ToolEntry]:
        """List all registered tools, optionally filtered by tags.

        Args:
            tags: If provided, return only tools that have ANY of the
                specified tags (union semantics).

        Returns:
            List of matching ToolEntry instances.
        """
        entries = self.list_items()
        if tags:
            tag_set = frozenset(tags)
            entries = [e for e in entries if e.tags & tag_set]
        return entries

    def get_names(self) -> list[str]:
        """Get all registered tool names.

        Returns:
            List of tool names.
        """
        return self.list_keys()

    # -- Config generation --

    def to_bot_config(self, name: str, **param_overrides: Any) -> dict[str, Any]:
        """Generate a bot config tool entry for the named tool.

        Returns a dict suitable for ``DynaBot._resolve_tool()``:
        ``{"class": "full.class.path", "params": {...}}``

        Args:
            name: Tool name to look up.
            **param_overrides: Override default params.

        Returns:
            Bot config dict for the tool.

        Raises:
            NotFoundError: If tool name is not registered.
        """
        entry = self.get(name)
        return entry.to_bot_config(**param_overrides)

    def to_bot_configs(
        self,
        names: Sequence[str],
        overrides: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate bot config entries for multiple tools.

        Args:
            names: Tool names to include.
            overrides: Per-tool param overrides keyed by tool name.

        Returns:
            List of bot config dicts.
        """
        overrides = overrides or {}
        return [
            self.to_bot_config(name, **overrides.get(name, {}))
            for name in names
        ]

    # -- Dependency validation --

    def get_requirements(self, names: Sequence[str]) -> frozenset[str]:
        """Get the union of all requirements for the given tool names.

        Args:
            names: Tool names to check.

        Returns:
            Set of all requirement identifiers across the named tools.
        """
        reqs: set[str] = set()
        for name in names:
            entry = self.get(name)
            reqs.update(entry.requires)
        return frozenset(reqs)

    def check_requirements(
        self,
        tool_names: Sequence[str],
        config: dict[str, Any],
    ) -> list[str]:
        """Check that tool requirements are satisfied by a config dict.

        Returns a list of warning messages for any unmet requirements.
        Tools with no requirements are always satisfied.

        Args:
            tool_names: Names of tools to check.
            config: Bot config dict to check against (top-level keys).

        Returns:
            List of warning strings (empty if all requirements met).
        """
        warnings: list[str] = []
        for name in tool_names:
            entry = self.get(name)
            for req in sorted(entry.requires):
                if req not in config:
                    warnings.append(
                        f"Tool '{name}' requires '{req}' "
                        f"but it is not configured"
                    )
        return warnings

    # -- Instantiation --

    def instantiate_tool(self, name: str, **param_overrides: Any) -> Any:
        """Import and instantiate a tool from its catalog entry.

        Uses ``resolve_callable()`` to import the class, then instantiates
        it with ``default_params`` merged with overrides. Prefers
        ``from_config()`` if the class defines it.

        Args:
            name: Tool name to instantiate.
            **param_overrides: Override default params.

        Returns:
            Instantiated tool.

        Raises:
            NotFoundError: If name not in catalog.
            ImportError: If class cannot be imported.
            ValueError: If resolved class is not callable.
        """
        from dataknobs_bots.tools.resolve import resolve_callable

        entry = self.get(name)
        tool_class = resolve_callable(entry.class_path)
        params = dict(entry.default_params)
        params.update(param_overrides)

        if hasattr(tool_class, "from_config") and callable(
            tool_class.from_config
        ):
            return tool_class.from_config(params)
        return tool_class(**params) if params else tool_class()

    def create_tool_registry(
        self,
        names: Sequence[str] | None = None,
        overrides: dict[str, dict[str, Any]] | None = None,
        strict: bool = False,
    ) -> Any:
        """Create a ToolRegistry populated from catalog entries.

        Imports and instantiates each named tool, registering them in a
        new ``ToolRegistry``.

        Args:
            names: Tool names to include (default: all registered).
            overrides: Per-tool param overrides keyed by tool name.
            strict: If True, raise on instantiation failure.
                If False (default), skip failed tools and log warnings.

        Returns:
            ToolRegistry with instantiated tools.
        """
        from dataknobs_llm.tools import ToolRegistry

        registry = ToolRegistry()
        target_names = list(names) if names else self.list_keys()
        overrides = overrides or {}

        for name in target_names:
            try:
                tool = self.instantiate_tool(name, **overrides.get(name, {}))
                registry.register_tool(tool)
            except Exception as e:
                if strict:
                    raise
                logger.warning(
                    "Failed to instantiate tool '%s': %s", name, e
                )

        return registry

    # -- Serialization --

    def to_dict(self) -> dict[str, Any]:
        """Serialize entire catalog to a dict (for YAML output).

        Returns:
            Dict with ``tools`` key containing list of tool dicts.
        """
        return {
            "tools": [entry.to_dict() for entry in self.list_items()]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCatalog:
        """Create a catalog from a dict (e.g., loaded from YAML).

        Args:
            data: Dict with ``tools`` key containing list of tool dicts.

        Returns:
            New ToolCatalog populated from the data.
        """
        catalog = cls()
        for tool_data in data.get("tools", []):
            catalog.register_from_dict(tool_data)
        return catalog


# -- Module-level singleton and factory --

default_catalog: ToolCatalog = ToolCatalog()
"""Module-level singleton catalog pre-populated with built-in tools."""


def _register_builtin_tools() -> None:
    """Register all built-in dataknobs-bots tools in the default catalog."""
    from dataknobs_bots.tools import (
        AddBankRecordTool,
        AddKBResourceTool,
        CheckKnowledgeSourceTool,
        FinalizeBankTool,
        GetTemplateDetailsTool,
        IngestKnowledgeBaseTool,
        KnowledgeSearchTool,
        ListAvailableToolsTool,
        ListBankRecordsTool,
        ListKBResourcesTool,
        ListTemplatesTool,
        PreviewConfigTool,
        RemoveBankRecordTool,
        RemoveKBResourceTool,
        SaveConfigTool,
        UpdateBankRecordTool,
        ValidateConfigTool,
    )

    for tool_class in [
        KnowledgeSearchTool,
        ListTemplatesTool,
        GetTemplateDetailsTool,
        PreviewConfigTool,
        ValidateConfigTool,
        SaveConfigTool,
        ListAvailableToolsTool,
        CheckKnowledgeSourceTool,
        ListKBResourcesTool,
        AddKBResourceTool,
        RemoveKBResourceTool,
        IngestKnowledgeBaseTool,
        ListBankRecordsTool,
        AddBankRecordTool,
        UpdateBankRecordTool,
        RemoveBankRecordTool,
        FinalizeBankTool,
    ]:
        default_catalog.register_from_class(tool_class)


_register_builtin_tools()


def create_default_catalog() -> ToolCatalog:
    """Create a new ToolCatalog pre-populated with built-in tools.

    Returns a fresh catalog (not the module-level singleton) so consumers
    can extend it without affecting other users of ``default_catalog``.

    Returns:
        New ToolCatalog with all built-in tools registered.
    """
    catalog = ToolCatalog()
    for entry in default_catalog.list_items():
        catalog.register_entry(entry)
    return catalog
