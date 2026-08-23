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

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

import yaml
from dataknobs_common.imports import resolve_callable
from dataknobs_common.paths import PathEscapeError
from dataknobs_llm.tools.context import ToolExecutionContext
from dataknobs_llm.tools.context_aware import ContextAwareTool

from dataknobs_bots.config.builder import DynaBotConfigBuilder
from dataknobs_bots.config.drafts import ConfigDraftManager
from dataknobs_bots.config.templates import ConfigTemplateRegistry
from dataknobs_bots.config.tool_catalog import InjectedCallable, injected_dependency
from dataknobs_bots.config.validation import ConfigValidator

logger = logging.getLogger(__name__)


def _registry_from_config(config: dict[str, Any]) -> ConfigTemplateRegistry:
    """Return the supplied template registry, or load one from disk.

    Shared by the two template tools, which declare the same dependency
    and had the same twelve lines each. An injected registry wins over
    ``template_dir``: it is the live one the deployment already has
    loaded, where the directory read is this tool's fallback for having
    been given nothing.

    Args:
        config: The params dict handed to ``from_config``.

    Returns:
        The registry to hand the tool. Empty when neither channel
        supplies one and ``template_dir`` does not exist.
    """
    injected = injected_dependency(config, "template_registry", ConfigTemplateRegistry)
    if injected is not None:
        return injected

    registry = ConfigTemplateRegistry()
    path = Path(config.get("template_dir", "configs/templates"))
    if path.is_dir():
        registry.load_from_directory(path)
    return registry


def _callable_from_config(config: dict[str, Any], key: str) -> Callable[..., Any] | None:
    """Return the callable under *key*, live or resolved from a dotted path.

    Both channels spell this key the same way, so unlike the object
    dependencies it needs a discriminator rather than a distinct name:
    a live callable is used as-is, a string goes to ``resolve_callable``.

    Args:
        config: The params dict handed to ``from_config``.
        key: The parameter name to read.

    Returns:
        The callable, or ``None`` when *key* is absent. Callers for whom
        the key is mandatory raise on the ``None``.

    Raises:
        DottedPathError: *key* holds a string that does not resolve.
    """
    live = injected_dependency(config, key, InjectedCallable)
    if live is not None:
        return live
    if key in config:
        return resolve_callable(config[key])
    return None


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


def _is_safe_config_name(name: str) -> bool:
    """Return True if ``name`` is a well-formed flat ``<name>.yaml`` filename.

    This is a **naming policy**, not the containment boundary. Containment
    is enforced where the path is composed, by
    :meth:`~dataknobs_bots.config.drafts.ConfigDraftManager.config_path`,
    which is what actually stops a write from leaving the output
    directory — and which reaches every caller, not only this tool.

    The policy exists on top of it because this name arrives from LLM tool
    arguments and user-driven wizard data: rejecting it here returns a
    structured ``{"success": False, "error": ...}`` the model can correct
    on its next turn, where the manager's guard raises. It is deliberately
    stricter than containment — a config name is flat, so a separator is
    refused even though ``team/alpha`` would be safely inside.
    """
    if not name or not name.strip() or name in (".", ".."):
        return False
    return not any(sep in name for sep in ("/", "\\", "\x00"))


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
            "description": ("List available bot configuration templates."),
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
        """Create from configuration.

        Args:
            config: Dict with either a live ``template_registry`` — the
                dependency this tool declares, injected by
                ``DynaBot._resolve_tool`` — or a ``template_dir`` key
                pointing at a directory of template YAML files.

        Returns:
            Configured ListTemplatesTool instance.
        """
        return cls(template_registry=_registry_from_config(config))

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
                    "required_variables": [v.name for v in t.get_required_variables()],
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
            "description": ("Get detailed information about a specific configuration template."),
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
        """Create from configuration.

        Args:
            config: Dict with either a live ``template_registry`` — the
                dependency this tool declares, injected by
                ``DynaBot._resolve_tool`` — or a ``template_dir`` key
                pointing at a directory of template YAML files.

        Returns:
            Configured GetTemplateDetailsTool instance.
        """
        return cls(template_registry=_registry_from_config(config))

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

    def missing_arguments_result(self, missing: list[str]) -> dict[str, Any]:
        """Report the omission alongside the names that would have worked.

        A model that omitted the template name is one turn from
        succeeding, and the registry contents are what close that turn.
        Same shape as the unknown-template branch below, so both
        failures read the same way.

        Args:
            missing: Declared-required names the caller did not supply.

        Returns:
            The base result plus the available template names.
        """
        result: dict[str, Any] = super().missing_arguments_result(missing)
        result["available"] = [t.name for t in self._registry.list_templates()]
        return result

    async def execute_with_context(
        self,
        context: ToolExecutionContext,
        template_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get template details.

        ``template_name`` is required by :attr:`schema` but optional in
        this signature, as it is for every tool in the package. A
        required positional parameter cannot be bound when the model
        omits it, so the call raises ``TypeError`` before
        :meth:`ContextAwareTool.execute` can report the omission -- the
        default is what lets the base class answer instead.

        The guard below covers the direct-call path only; through
        ``execute`` the base class has already returned.

        Args:
            context: Execution context.
            template_name: Name of the template.

        Returns:
            Dict with template details, or error if missing or not found.
        """
        if template_name is None:
            return self.missing_arguments_result(["template_name"])

        template = self._registry.get(template_name)
        if template is None:
            return {
                "error": f"Template not found: {template_name}",
                "available": [t.name for t in self._registry.list_templates()],
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
            "required_variables": [v.to_dict() for v in template.get_required_variables()],
            "optional_variables": [v.to_dict() for v in template.get_optional_variables()],
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
                "Preview the bot configuration being built from the current wizard data."
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
        """Create from configuration.

        Args:
            config: Dict with a required ``builder_factory`` key — the
                live callable, injected by ``DynaBot._resolve_tool`` for
                the dependency this tool declares, or a dotted import
                path to one. Either way it accepts wizard data and
                returns a ``DynaBotConfigBuilder``.

        Returns:
            Configured PreviewConfigTool instance.

        Raises:
            KeyError: No ``builder_factory`` was supplied.
        """
        factory = _callable_from_config(config, "builder_factory")
        if factory is None:
            # Mandatory for this tool. Same exception a caller omitting
            # it has always seen, when the body indexed the dict.
            raise KeyError("builder_factory")
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
            config = builder.build_unvalidated()
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

    When a ``builder_factory`` is supplied the builder's own validator
    decides the verdict — the same validator ``build`` and
    ``build_portable`` run — so wiring this tool and ``SaveConfigTool``
    to **the same factory** makes the save outcome predictable from the
    validate outcome, at either setting of ``portable``.

    Two things fall outside that guarantee, both by construction:

    - Each tool resolves its own ``builder_factory`` from its own config
      block. Nothing checks that the two name the same callable, and two
      different builders are entitled to two different verdicts.
    - An explicitly supplied ``validator`` runs in addition to the
      builder's, so this tool can refuse what save would accept. That
      direction is deliberate. The failure modes are not symmetric: an
      extra error stops an author, a missing one misleads them.

    Attributes:
        _validator: Optional additional ConfigValidator.
        _builder_factory: Optional factory for building config from wizard data.
    """

    @classmethod
    def catalog_metadata(cls) -> dict[str, Any]:
        """Return catalog metadata for this tool class."""
        return {
            "name": "validate_config",
            "description": ("Validate the bot configuration being built."),
            "tags": ("configbot",),
            # The verdict comes from the builder's validator, so the
            # factory that produces the builder is what this tool needs
            # supplied. A `validator` is optional and additive.
            "requires": ("builder_factory",),
        }

    def __init__(
        self,
        validator: ConfigValidator | None = None,
        builder_factory: Callable[[dict[str, Any]], DynaBotConfigBuilder] | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            validator: Optional additional validator. When a
                ``builder_factory`` is supplied the builder's own
                validator is authoritative and this one runs in
                addition to it, contributing extra errors only.
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
        """Create from configuration.

        No ``validator`` is constructed here. One built from no schema
        is the construct that let this tool contradict ``save_config``,
        and supplying it unconditionally would run it as a second pass
        over every YAML-wired instance — the "wire it twice and keep the
        two in sync" this tool exists to avoid. A consumer that wants an
        additional validator passes one to ``__init__``.

        Args:
            config: Dict with an optional ``builder_factory`` key — the
                live callable, injected by ``DynaBot._resolve_tool`` for
                the dependency this tool declares, or a dotted import
                path to one.

        Returns:
            Configured ValidateConfigTool instance.
        """
        return cls(builder_factory=_callable_from_config(config, "builder_factory"))

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
            except Exception as e:
                return {
                    "valid": False,
                    "errors": [f"Failed to build configuration: {e}"],
                }
            try:
                # The builder's own validator is authoritative. It is the one
                # `build()` and `build_portable()` run, so anything else here
                # can report a verdict the save path contradicts -- which is
                # precisely what a schema-less `ConfigValidator()` used to do.
                result = builder.validate()
            except Exception as e:
                return {
                    "valid": False,
                    "errors": [f"Failed to validate configuration: {e}"],
                }

            if self._validator is not None:
                try:
                    # An explicitly supplied validator runs *in addition*,
                    # never instead: it can only add errors, so it cannot
                    # reintroduce the disagreement. `merge_unique` because
                    # both validators run `validate_completeness` over the
                    # same config and would otherwise report each shared
                    # failure twice.
                    extra = self._validator.validate(builder.build_unvalidated())
                except Exception as e:
                    return {
                        "valid": False,
                        "errors": [f"Failed to validate configuration: {e}"],
                    }
                result = result.merge_unique(extra)
        else:
            validator = self._validator or ConfigValidator()
            result = validator.validate(wizard_data)

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
    used instead of ``build()``, producing a config with a
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
            "description": ("Save and finalize the bot configuration."),
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
                siblings). When False (default), use ``build()`` for
                flat format. Both validate.
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
        """Create from configuration.

        Each dependency below may arrive as the live object instead of
        its YAML form: ``draft_manager`` because
        ``DynaBot._resolve_tool`` injects the dependency this tool
        declares, and the two callables because
        ``ToolCatalog.instantiate_tool`` passes its keywords straight
        into this dict whatever the catalog declares.

        Args:
            config: Dict with keys:
                - ``draft_manager`` (ConfigDraftManager, optional): The
                  live manager, in place of ``config_dir``.
                - ``config_dir`` (str): Output directory for configs.
                - ``builder_factory`` (callable | str, optional).
                - ``on_save`` (callable | str, optional).
                - ``portable`` (bool, optional): Use portable output format.

        Returns:
            Configured SaveConfigTool instance.
        """
        manager = injected_dependency(config, "draft_manager", ConfigDraftManager)
        if manager is None:
            manager = ConfigDraftManager(output_dir=Path(config.get("config_dir", "configs")))

        return cls(
            draft_manager=manager,
            on_save=_callable_from_config(config, "on_save"),
            builder_factory=_callable_from_config(config, "builder_factory"),
            portable=config.get("portable", False),
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
        if not _is_safe_config_name(name):
            return {
                "success": False,
                "error": (
                    f"Invalid config name '{name}': must not contain path "
                    "separators or path-traversal segments"
                ),
            }

        # Build final config. Both branches validate and raise: the flag
        # selects the output shape, not whether the config is checked.
        # `build_unvalidated()` skips validation, and reaching for it
        # here is what let this tool write to disk the config
        # `validate_config` had just refused -- on the flag's default.
        if self._builder_factory is not None:
            try:
                builder = self._builder_factory(wizard_data)
                config = builder.build_portable() if self._portable else builder.build()
            except Exception as e:
                return {"success": False, "error": f"Failed to build configuration: {e}"}
        else:
            config = {k: v for k, v in wizard_data.items() if not k.startswith("_")}

        # Finalizing the draft and writing the config are blocking disk I/O;
        # offload the whole persist tail so the tool never stalls the loop.
        draft_id = wizard_data.get("_draft_id")
        try:
            final_path = await asyncio.to_thread(self._persist_config, name, draft_id, config)
        except PathEscapeError as e:
            # The entry-point check above covers ``name``. ``draft_id``
            # comes from wizard data and reaches the manager unchecked,
            # so the manager's guard is what catches it — and it raises,
            # where every other refusal in this tool returns. Translating
            # it here keeps one contract: the model gets an error it can
            # correct on its next turn instead of a tool-call crash.
            logger.warning(
                "Refused to save configuration %r: %s",
                name,
                e,
                extra={"config_name": name, "conversation_id": context.conversation_id},
            )
            return {"success": False, "error": str(e)}

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
                self._on_save(name, config)
            except Exception:
                logger.exception("on_save callback failed for '%s'", name)

        return {
            "success": True,
            "config_name": name,
            "file_path": str(final_path),
            "activated": activate,
        }

    def _persist_config(
        self, name: str, draft_id: str | None, final_config: dict[str, Any]
    ) -> Path:
        """Finalize any draft and write the config to disk.

        Synchronous, blocking disk I/O (draft finalize, ``mkdir``, YAML
        write); :meth:`execute_with_context` runs it via
        :func:`asyncio.to_thread` so the event loop is never blocked.

        Raises:
            PathEscapeError: If ``name`` or ``draft_id`` addresses a file
                outside the draft manager's output directory. Resolved
                before anything is written, so an escaping name leaves no
                partial state. :meth:`execute_with_context` translates it
                into this tool's structured error rather than letting it
                surface as a raise.
        """
        # Resolve before writing anything. This method used to compose the
        # path itself and re-check it afterwards, which guarded its own
        # open() but not the finalize() below — that write went through
        # the manager, where the check could not see it.
        final_path = self._draft_manager.config_path(name)

        # Finalize cleans up the draft file, but we always write the
        # freshly-built config (the draft may be stale).
        if draft_id:
            try:
                self._draft_manager.finalize(draft_id, final_name=name)
            except FileNotFoundError:
                logger.warning("Draft %s not found, saving directly", draft_id)

        # ``parents=True`` creates the output dir itself when the name is
        # flat, so this is the only mkdir needed for either shape.
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with open(final_path, "w") as f:
            yaml.dump(final_config, f, default_flow_style=False, sort_keys=False)
        return final_path


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
            "description": ("List tools that can be added to the bot configuration."),
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
                        "Optional category to filter tools by. Omit to list all tools."
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
            filtered = [t for t in self._tools if t.get("category", "").lower() == category.lower()]
        else:
            filtered = list(self._tools)

        categories = sorted({t["category"] for t in self._tools if "category" in t})

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
            summary["sections"].append({"name": "LLM", "value": f"$resource: {llm['$resource']}"})
        else:
            provider = llm.get("provider", "unknown")
            model = llm.get("model", "default")
            summary["sections"].append({"name": "LLM", "value": f"{provider}/{model}"})

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
        summary["sections"].append({"name": "Memory", "value": memory.get("type", "default")})

    reasoning = config.get("reasoning", {})
    if isinstance(reasoning, dict) and reasoning:
        summary["sections"].append(
            {"name": "Reasoning", "value": reasoning.get("strategy", "simple")}
        )

    kb = config.get("knowledge_base", {})
    if isinstance(kb, dict) and kb.get("enabled"):
        summary["sections"].append({"name": "Knowledge Base", "value": "enabled"})

    tools = config.get("tools", [])
    if isinstance(tools, list) and tools:
        summary["sections"].append({"name": "Tools", "value": f"{len(tools)} configured"})

    prompt = config.get("system_prompt")
    if prompt:
        if isinstance(prompt, str):
            summary["sections"].append({"name": "System Prompt", "value": f"{len(prompt)} chars"})
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
