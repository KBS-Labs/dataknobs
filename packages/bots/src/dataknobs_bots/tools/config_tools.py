"""ConfigBot toolkit tools for DynaBot configuration workflows.

Provides reusable LLM-callable tools for wizard-driven bot configuration.
Each tool follows the ContextAwareTool pattern with static dependencies
injected via constructor and dynamic context via ToolExecutionContext.

Tools:
- ListTemplatesTool: List available configuration templates
- GetTemplateDetailsTool: Get details for a specific template
- PreviewConfigTool: Preview the current configuration being built
- ValidateConfigTool: Validate the current configuration
- SaveConfigTool: Finalize and save the configuration
- ListAvailableToolsTool: List tools available for bot configuration

Example:
    ```python
    from dataknobs_bots.config.templates import ConfigTemplateRegistry
    from dataknobs_bots.config.drafts import ConfigDraftManager
    from dataknobs_bots.config.validation import ConfigValidator
    from dataknobs_bots.tools.config_tools import (
        ListTemplatesTool, PreviewConfigTool, SaveConfigTool,
    )

    registry = ConfigTemplateRegistry()
    registry.load_from_directory(Path("configs/templates"))

    list_tool = ListTemplatesTool(template_registry=registry)
    preview_tool = PreviewConfigTool(builder_factory=my_builder_factory)
    save_tool = SaveConfigTool(draft_manager=manager, on_save=my_callback)
    ```
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import yaml
from dataknobs_llm.tools.context import ToolExecutionContext
from dataknobs_llm.tools.context_aware import ContextAwareTool

from dataknobs_bots.config.builder import DynaBotConfigBuilder
from dataknobs_bots.config.drafts import ConfigDraftManager
from dataknobs_bots.config.templates import ConfigTemplateRegistry
from dataknobs_bots.config.validation import ConfigValidator

logger = logging.getLogger(__name__)


def _get_wizard_data(context: ToolExecutionContext) -> dict[str, Any]:
    """Extract wizard collected data from tool execution context (copy).

    Returns a shallow copy of the wizard's collected data. Use this for
    read-only access to wizard data.

    Args:
        context: The tool execution context.

    Returns:
        A copy of the wizard's collected data dict, or empty dict if unavailable.
    """
    if context.wizard_state and context.wizard_state.collected_data:
        return dict(context.wizard_state.collected_data)
    return {}


def _get_wizard_data_ref(context: ToolExecutionContext) -> dict[str, Any]:
    """Get a mutable reference to wizard collected data.

    Returns the original collected_data dict (not a copy) for tools that
    need to mutate wizard state (e.g., KB tools that add/remove resources).

    Uses ``is not None`` rather than truthiness so that an empty dict
    (new wizard session) is still returned by reference.

    Args:
        context: The tool execution context.

    Returns:
        The wizard's collected data dict reference, or empty dict if unavailable.
    """
    if context.wizard_state and context.wizard_state.collected_data is not None:
        return context.wizard_state.collected_data
    return {}


class ListTemplatesTool(ContextAwareTool):
    """Tool for listing available configuration templates.

    Allows the LLM to discover what templates are available,
    optionally filtered by tags.

    Attributes:
        _registry: Template registry to query.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "list_templates",
            "description": (
                "List available bot configuration templates."
            ),
            "tags": ("configbot",),
            "requires": ("template_registry",),
            "default_params": {"template_dir": "configs/templates"},
        }

    def __init__(self, template_registry: ConfigTemplateRegistry) -> None:
        """Initialize the tool.

        Args:
            template_registry: Registry containing available templates.
        """
        super().__init__(
            name="list_templates",
            description=(
                "List available bot configuration templates. "
                "Optionally filter by tags to find templates for "
                "specific use cases."
            ),
        )
        self._registry = template_registry

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ListTemplatesTool:
        """Create from YAML-compatible configuration.

        Args:
            config: Dict with ``template_dir`` key pointing to a
                directory containing template YAML files.

        Returns:
            Configured ListTemplatesTool instance.
        """
        from pathlib import Path

        template_dir = config.get("template_dir", "configs/templates")
        registry = ConfigTemplateRegistry()
        path = Path(template_dir)
        if path.is_dir():
            registry.load_from_directory(path)
        return cls(template_registry=registry)

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags to filter templates by",
                },
            },
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """List available templates.

        Args:
            context: Execution context.
            tags: Optional tags to filter by.

        Returns:
            Dict with list of template summaries.
        """
        templates = self._registry.list_templates(tags=tags)

        logger.debug(
            "Listed %d templates (tags=%s)",
            len(templates),
            tags,
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "templates": [
                {
                    "name": t.name,
                    "description": t.description,
                    "version": t.version,
                    "tags": t.tags,
                    "variables_count": len(t.variables),
                    "required_variables": [
                        v.name for v in t.get_required_variables()
                    ],
                }
                for t in templates
            ],
            "count": len(templates),
        }


class GetTemplateDetailsTool(ContextAwareTool):
    """Tool for getting detailed information about a template.

    Returns the full template definition including all variables,
    their types, defaults, and constraints.

    Attributes:
        _registry: Template registry to query.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "get_template_details",
            "description": (
                "Get detailed information about a specific "
                "configuration template."
            ),
            "tags": ("configbot",),
            "requires": ("template_registry",),
            "default_params": {"template_dir": "configs/templates"},
        }

    def __init__(self, template_registry: ConfigTemplateRegistry) -> None:
        """Initialize the tool.

        Args:
            template_registry: Registry containing available templates.
        """
        super().__init__(
            name="get_template_details",
            description=(
                "Get detailed information about a specific configuration "
                "template, including all variables and their requirements."
            ),
        )
        self._registry = template_registry

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GetTemplateDetailsTool:
        """Create from YAML-compatible configuration.

        Args:
            config: Dict with ``template_dir`` key pointing to a
                directory containing template YAML files.

        Returns:
            Configured GetTemplateDetailsTool instance.
        """
        from pathlib import Path

        template_dir = config.get("template_dir", "configs/templates")
        registry = ConfigTemplateRegistry()
        path = Path(template_dir)
        if path.is_dir():
            registry.load_from_directory(path)
        return cls(template_registry=registry)

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "template_name": {
                    "type": "string",
                    "description": "Name of the template to get details for",
                },
            },
            "required": ["template_name"],
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        template_name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get template details.

        Args:
            context: Execution context.
            template_name: Name of the template.

        Returns:
            Dict with template details, or error if not found.
        """
        template = self._registry.get(template_name)
        if template is None:
            return {
                "error": f"Template not found: {template_name}",
                "available": [
                    t.name for t in self._registry.list_templates()
                ],
            }

        logger.debug(
            "Retrieved template details: %s",
            template_name,
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "name": template.name,
            "description": template.description,
            "version": template.version,
            "tags": template.tags,
            "variables": [v.to_dict() for v in template.variables],
            "required_variables": [
                v.to_dict() for v in template.get_required_variables()
            ],
            "optional_variables": [
                v.to_dict() for v in template.get_optional_variables()
            ],
        }


class PreviewConfigTool(ContextAwareTool):
    """Tool for previewing the configuration being built.

    Uses a consumer-provided ``builder_factory`` to construct the
    configuration from wizard data. This is the key extension point:
    the factory encapsulates domain-specific logic.

    Attributes:
        _builder_factory: Callable that creates a configured builder
            from wizard data.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "preview_config",
            "description": (
                "Preview the bot configuration being built from "
                "the current wizard data."
            ),
            "tags": ("configbot",),
            "requires": ("builder_factory",),
        }

    def __init__(
        self,
        builder_factory: Callable[[dict[str, Any]], DynaBotConfigBuilder],
    ) -> None:
        """Initialize the tool.

        Args:
            builder_factory: Function that takes wizard collected data
                and returns a configured DynaBotConfigBuilder. This is
                where consumers inject domain-specific config logic.
        """
        super().__init__(
            name="preview_config",
            description=(
                "Preview the bot configuration being built from the "
                "current wizard data. Shows what the final config will "
                "look like."
            ),
        )
        self._builder_factory = builder_factory

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PreviewConfigTool:
        """Create from YAML-compatible configuration.

        Args:
            config: Dict with ``builder_factory`` key — a dotted
                import path to a callable that accepts wizard data
                and returns a ``DynaBotConfigBuilder``.

        Returns:
            Configured PreviewConfigTool instance.
        """
        from .resolve import resolve_callable

        factory_ref = config["builder_factory"]
        factory = resolve_callable(factory_ref)
        return cls(builder_factory=factory)

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Output format: 'summary', 'full', or 'yaml'",
                    "enum": ["summary", "full", "yaml"],
                    "default": "summary",
                },
            },
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        format: str = "summary",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Preview the current configuration.

        Args:
            context: Execution context with wizard state.
            format: Output format ('summary', 'full', or 'yaml').

        Returns:
            Dict with the configuration preview.
        """
        wizard_data = _get_wizard_data(context)
        if not wizard_data:
            return {"error": "No wizard data available for preview"}

        try:
            builder = self._builder_factory(wizard_data)
            config = builder._build_internal()
        except Exception as e:
            logger.exception("Failed to build config for preview")
            return {"error": f"Failed to build configuration: {e}"}

        logger.debug(
            "Generated config preview (format=%s)",
            format,
            extra={"conversation_id": context.conversation_id},
        )

        if format == "yaml":
            return {"yaml": yaml.dump(config, default_flow_style=False, sort_keys=False)}
        elif format == "full":
            return {"config": config}
        else:
            return _build_summary(config)


class ValidateConfigTool(ContextAwareTool):
    """Tool for validating the current configuration.

    Runs the full validation pipeline and returns errors and warnings.

    Attributes:
        _validator: ConfigValidator instance.
        _builder_factory: Optional factory for building config from wizard data.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "validate_config",
            "description": (
                "Validate the bot configuration being built."
            ),
            "tags": ("configbot",),
            "requires": ("validator",),
        }

    def __init__(
        self,
        validator: ConfigValidator,
        builder_factory: Callable[[dict[str, Any]], DynaBotConfigBuilder] | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            validator: Validator to use for checking configs.
            builder_factory: Optional factory to build config from wizard
                data before validation. If not provided, validates the
                raw wizard data as a config dict.
        """
        super().__init__(
            name="validate_config",
            description=(
                "Validate the bot configuration being built. "
                "Checks for completeness, schema compliance, and "
                "portability issues."
            ),
        )
        self._validator = validator
        self._builder_factory = builder_factory

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ValidateConfigTool:
        """Create from YAML-compatible configuration.

        Args:
            config: Dict with optional ``builder_factory`` key — a
                dotted import path to a callable.

        Returns:
            Configured ValidateConfigTool instance.
        """
        validator = ConfigValidator()
        factory = None
        if "builder_factory" in config:
            from .resolve import resolve_callable

            factory = resolve_callable(config["builder_factory"])
        return cls(validator=validator, builder_factory=factory)

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
        """Validate the current configuration.

        Args:
            context: Execution context with wizard state.

        Returns:
            Dict with validation results.
        """
        wizard_data = _get_wizard_data(context)
        if not wizard_data:
            return {"valid": False, "errors": ["No wizard data available"]}

        if self._builder_factory is not None:
            try:
                builder = self._builder_factory(wizard_data)
                config = builder._build_internal()
            except Exception as e:
                return {
                    "valid": False,
                    "errors": [f"Failed to build configuration: {e}"],
                }
        else:
            config = wizard_data

        result = self._validator.validate(config)

        logger.debug(
            "Validated config: valid=%s, errors=%d, warnings=%d",
            result.valid,
            len(result.errors),
            len(result.warnings),
            extra={"conversation_id": context.conversation_id},
        )

        return result.to_dict()


class SaveConfigTool(ContextAwareTool):
    """Tool for saving/finalizing the configuration.

    Finalizes the draft and writes the final config file. Optionally
    calls a consumer-provided callback for post-save actions (e.g.,
    registering the bot with a manager).

    When ``portable=True``, the builder's ``build_portable()`` method is
    used instead of ``_build_internal()``, producing a config with a
    ``bot`` wrapper key suitable for environment-aware deployment.

    Attributes:
        _draft_manager: Draft manager for file operations.
        _on_save: Optional callback invoked after successful save.
        _builder_factory: Optional factory for building config from wizard data.
        _portable: Whether to use portable (bot-wrapped) output format.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "save_config",
            "description": (
                "Save and finalize the bot configuration."
            ),
            "tags": ("configbot",),
            "requires": ("draft_manager",),
        }

    def __init__(
        self,
        draft_manager: ConfigDraftManager,
        on_save: Callable[[str, dict[str, Any]], Any] | None = None,
        builder_factory: Callable[[dict[str, Any]], DynaBotConfigBuilder] | None = None,
        portable: bool = False,
    ) -> None:
        """Initialize the tool.

        Args:
            draft_manager: Manager for draft file operations.
            on_save: Optional callback called with (config_name, config)
                after successful save. Can be used for post-save actions
                like bot registration.
            builder_factory: Optional factory to build final config from
                wizard data before saving.
            portable: When True, use ``build_portable()`` for output
                (wraps config under ``bot`` key with custom sections as
                siblings). When False (default), use ``_build_internal()``
                for flat format.
        """
        super().__init__(
            name="save_config",
            description=(
                "Save and finalize the bot configuration. Writes the "
                "final config file and optionally activates the bot."
            ),
        )
        self._draft_manager = draft_manager
        self._on_save = on_save
        self._builder_factory = builder_factory
        self._portable = portable

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SaveConfigTool:
        """Create from YAML-compatible configuration.

        Args:
            config: Dict with keys:
                - ``config_dir`` (str): Output directory for configs.
                - ``builder_factory`` (str, optional): Dotted import path.
                - ``on_save`` (str, optional): Dotted import path.
                - ``portable`` (bool, optional): Use portable output format.

        Returns:
            Configured SaveConfigTool instance.
        """
        from pathlib import Path

        config_dir = config.get("config_dir", "configs")
        manager = ConfigDraftManager(output_dir=Path(config_dir))

        on_save = None
        factory = None
        if "on_save" in config or "builder_factory" in config:
            from .resolve import resolve_callable

            if "on_save" in config:
                on_save = resolve_callable(config["on_save"])
            if "builder_factory" in config:
                factory = resolve_callable(config["builder_factory"])

        portable = config.get("portable", False)
        return cls(
            draft_manager=manager,
            on_save=on_save,
            builder_factory=factory,
            portable=portable,
        )

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "config_name": {
                    "type": "string",
                    "description": "Name for the saved configuration file",
                },
                "activate": {
                    "type": "boolean",
                    "description": "Whether to activate the bot after saving",
                    "default": False,
                },
            },
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        config_name: str | None = None,
        activate: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Save the configuration.

        Args:
            context: Execution context with wizard state.
            config_name: Name for the config file.
            activate: Whether to activate the bot.

        Returns:
            Dict with save result (success, file path, etc.).
        """
        wizard_data = _get_wizard_data(context)
        if not wizard_data:
            return {"success": False, "error": "No wizard data available"}

        # Determine config name
        name = config_name or wizard_data.get("domain_id") or wizard_data.get("config_name")
        if not name:
            return {
                "success": False,
                "error": "No config_name provided and no domain_id in wizard data",
            }

        # Build final config
        if self._builder_factory is not None:
            try:
                builder = self._builder_factory(wizard_data)
                if self._portable:
                    config = builder.build_portable()
                else:
                    config = builder._build_internal()
            except Exception as e:
                return {"success": False, "error": f"Failed to build configuration: {e}"}
        else:
            config = {
                k: v for k, v in wizard_data.items() if not k.startswith("_")
            }

        # Check for existing draft — finalize cleans up the draft file,
        # but we always use the freshly-built config (draft may be stale)
        draft_id = wizard_data.get("_draft_id")
        if draft_id:
            try:
                self._draft_manager.finalize(draft_id, final_name=name)
            except FileNotFoundError:
                logger.warning("Draft %s not found, saving directly", draft_id)
        final_config = config

        # Write the final file
        output_dir = self._draft_manager.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / f"{name}.yaml"
        with open(final_path, "w") as f:
            yaml.dump(final_config, f, default_flow_style=False, sort_keys=False)

        logger.info(
            "Saved configuration '%s' to %s",
            name,
            final_path,
            extra={
                "config_name": name,
                "activate": activate,
                "conversation_id": context.conversation_id,
            },
        )

        # Run consumer callback
        if self._on_save is not None:
            try:
                self._on_save(name, final_config)
            except Exception:
                logger.exception("on_save callback failed for '%s'", name)

        return {
            "success": True,
            "config_name": name,
            "file_path": str(final_path),
            "activated": activate,
        }


class ListAvailableToolsTool(ContextAwareTool):
    """Tool for listing tools available to configure for a bot.

    Takes a constructor-injected catalog of available tools and lets the
    LLM browse them, optionally filtering by category. The catalog data
    is consumer-specific — each DynaBot consumer provides its own list.

    Attributes:
        _tools: The available tool catalog.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "list_available_tools",
            "description": (
                "List tools that can be added to the bot configuration."
            ),
            "tags": ("configbot",),
        }

    def __init__(self, available_tools: list[dict[str, Any]]) -> None:
        """Initialize the tool.

        Args:
            available_tools: List of tool descriptors. Each dict should
                have at minimum ``name`` and ``description`` keys.
                Optional: ``category``, ``params``, ``class``.
        """
        super().__init__(
            name="list_available_tools",
            description=(
                "List tools that can be added to the bot configuration. "
                "Optionally filter by category."
            ),
        )
        self._tools = available_tools

    @property
    def schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Optional category to filter tools by. "
                        "Omit to list all tools."
                    ),
                },
            },
        }

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        category: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """List available tools.

        Args:
            context: Execution context.
            category: Optional category to filter by.

        Returns:
            Dict with matching tools, count, and available categories.
        """
        if category:
            filtered = [
                t for t in self._tools
                if t.get("category", "").lower() == category.lower()
            ]
        else:
            filtered = list(self._tools)

        categories = sorted({
            t["category"] for t in self._tools if "category" in t
        })

        logger.debug(
            "Listed %d available tools (category=%s)",
            len(filtered),
            category,
            extra={"conversation_id": context.conversation_id},
        )

        return {
            "tools": filtered,
            "count": len(filtered),
            "categories": categories,
        }


def _build_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Build a human-readable summary of a configuration.

    Args:
        config: Configuration dictionary to summarize.

    Returns:
        Summary dict with key settings.
    """
    summary: dict[str, Any] = {"sections": []}

    llm = config.get("llm", {})
    if isinstance(llm, dict):
        if "$resource" in llm:
            summary["sections"].append(
                {"name": "LLM", "value": f"$resource: {llm['$resource']}"}
            )
        else:
            provider = llm.get("provider", "unknown")
            model = llm.get("model", "default")
            summary["sections"].append(
                {"name": "LLM", "value": f"{provider}/{model}"}
            )

    storage = config.get("conversation_storage", {})
    if isinstance(storage, dict):
        if "$resource" in storage:
            summary["sections"].append(
                {"name": "Storage", "value": f"$resource: {storage['$resource']}"}
            )
        else:
            summary["sections"].append(
                {"name": "Storage", "value": storage.get("backend", "unknown")}
            )

    memory = config.get("memory", {})
    if isinstance(memory, dict) and memory:
        summary["sections"].append(
            {"name": "Memory", "value": memory.get("type", "default")}
        )

    reasoning = config.get("reasoning", {})
    if isinstance(reasoning, dict) and reasoning:
        summary["sections"].append(
            {"name": "Reasoning", "value": reasoning.get("strategy", "simple")}
        )

    kb = config.get("knowledge_base", {})
    if isinstance(kb, dict) and kb.get("enabled"):
        summary["sections"].append(
            {"name": "Knowledge Base", "value": "enabled"}
        )

    tools = config.get("tools", [])
    if isinstance(tools, list) and tools:
        summary["sections"].append(
            {"name": "Tools", "value": f"{len(tools)} configured"}
        )

    prompt = config.get("system_prompt")
    if prompt:
        if isinstance(prompt, str):
            summary["sections"].append(
                {"name": "System Prompt", "value": f"{len(prompt)} chars"}
            )
        elif isinstance(prompt, dict):
            if "name" in prompt:
                summary["sections"].append(
                    {"name": "System Prompt", "value": f"template: {prompt['name']}"}
                )
            else:
                content = prompt.get("content", "")
                summary["sections"].append(
                    {"name": "System Prompt", "value": f"{len(content)} chars"}
                )

    return summary
