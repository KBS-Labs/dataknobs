"""Fluent builder for DynaBot configurations.

Provides a composable, extensible builder for creating DynaBot configurations.
Supports both flat format (for ``DynaBot.from_config()``) and portable format
(with ``$resource`` references for environment-aware deployment).

Example:
    ```python
    from dataknobs_bots.config.builder import DynaBotConfigBuilder

    config = (
        DynaBotConfigBuilder()
        .set_llm("ollama", model="llama3.2")
        .set_conversation_storage("memory")
        .set_system_prompt(content="You are a helpful assistant.")
        .set_memory("buffer", max_messages=50)
        .build()
    )
    # config is compatible with DynaBot.from_config()
    ```
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from typing_extensions import Self

import yaml

from .schema import DynaBotConfigSchema
from .templates import ConfigTemplate
from .validation import ConfigValidator, ValidationResult

logger = logging.getLogger(__name__)


class DynaBotConfigBuilder:
    """Fluent builder for DynaBot configurations.

    Provides setter methods for each DynaBot component that return ``self``
    for method chaining. Consumer-specific sections are added via
    ``set_custom_section()``.

    Two output formats:
    - ``build()`` returns flat format compatible with ``DynaBot.from_config()``
    - ``build_portable()`` returns environment-aware format with ``$resource``
      references and a ``bot`` wrapper key
    """

    def __init__(self, schema: DynaBotConfigSchema | None = None) -> None:
        """Initialize the builder.

        Args:
            schema: Optional schema for validation. If not provided, a
                default schema is created.
        """
        self._schema = schema or DynaBotConfigSchema()
        self._config: dict[str, Any] = {}
        self._custom_sections: dict[str, Any] = {}
        self._validator = ConfigValidator(self._schema)

    # -- LLM --

    def set_llm(
        self,
        provider: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Set the LLM provider configuration (flat/direct format).

        Args:
            provider: LLM provider name (e.g., 'ollama', 'openai').
            model: Model name or identifier.
            **kwargs: Additional provider-specific settings
                (temperature, max_tokens, etc.).

        Returns:
            self for method chaining.
        """
        llm_config: dict[str, Any] = {"provider": provider}
        if model is not None:
            llm_config["model"] = model
        llm_config.update(kwargs)
        self._config["llm"] = llm_config
        return self

    def set_llm_resource(
        self,
        resource_name: str = "default",
        resource_type: str = "llm_providers",
        **overrides: Any,
    ) -> Self:
        """Set the LLM configuration using a $resource reference.

        Args:
            resource_name: Resource name to resolve at runtime.
            resource_type: Resource type category.
            **overrides: Override values applied after resolution.

        Returns:
            self for method chaining.
        """
        llm_config: dict[str, Any] = {
            "$resource": resource_name,
            "type": resource_type,
        }
        llm_config.update(overrides)
        self._config["llm"] = llm_config
        return self

    # -- Conversation Storage --

    def set_conversation_storage(
        self,
        backend: str,
        **kwargs: Any,
    ) -> Self:
        """Set the conversation storage backend (flat/direct format).

        Args:
            backend: Storage backend name (e.g., 'memory', 'sqlite').
            **kwargs: Additional backend-specific settings.

        Returns:
            self for method chaining.
        """
        storage_config: dict[str, Any] = {"backend": backend}
        storage_config.update(kwargs)
        self._config["conversation_storage"] = storage_config
        return self

    def set_conversation_storage_resource(
        self,
        resource_name: str = "conversations",
        resource_type: str = "databases",
        **overrides: Any,
    ) -> Self:
        """Set conversation storage using a $resource reference.

        Args:
            resource_name: Resource name to resolve at runtime.
            resource_type: Resource type category.
            **overrides: Override values applied after resolution.

        Returns:
            self for method chaining.
        """
        storage_config: dict[str, Any] = {
            "$resource": resource_name,
            "type": resource_type,
        }
        storage_config.update(overrides)
        self._config["conversation_storage"] = storage_config
        return self

    # -- Components --

    def set_memory(self, memory_type: str, **kwargs: Any) -> Self:
        """Set the memory configuration.

        Args:
            memory_type: Memory type (e.g., 'buffer', 'vector').
            **kwargs: Additional memory settings (max_messages, etc.).

        Returns:
            self for method chaining.
        """
        memory_config: dict[str, Any] = {"type": memory_type}
        memory_config.update(kwargs)
        self._config["memory"] = memory_config
        return self

    def set_reasoning(self, strategy: str, **kwargs: Any) -> Self:
        """Set the reasoning strategy.

        Args:
            strategy: Reasoning strategy (e.g., 'simple', 'react', 'wizard').
            **kwargs: Additional strategy settings.

        Returns:
            self for method chaining.
        """
        reasoning_config: dict[str, Any] = {"strategy": strategy}
        reasoning_config.update(kwargs)
        self._config["reasoning"] = reasoning_config
        return self

    def set_system_prompt(
        self,
        content: str | None = None,
        name: str | None = None,
        rag_configs: list[dict[str, Any]] | None = None,
    ) -> Self:
        """Set the system prompt configuration.

        Provide either ``content`` (inline prompt) or ``name`` (template
        reference). Optionally add RAG configurations for prompt enhancement.

        Args:
            content: Inline prompt content.
            name: Prompt template name.
            rag_configs: RAG configurations for prompt enhancement.

        Returns:
            self for method chaining.
        """
        if content is not None and name is None and rag_configs is None:
            self._config["system_prompt"] = content
        else:
            prompt_config: dict[str, Any] = {}
            if content is not None:
                prompt_config["content"] = content
            if name is not None:
                prompt_config["name"] = name
            if rag_configs is not None:
                prompt_config["rag_configs"] = rag_configs
            self._config["system_prompt"] = prompt_config
        return self

    def set_knowledge_base(self, **kwargs: Any) -> Self:
        """Set the knowledge base configuration.

        Args:
            **kwargs: Knowledge base settings (enabled, type,
                vector_store, embedding, etc.).

        Returns:
            self for method chaining.
        """
        self._config["knowledge_base"] = dict(kwargs)
        return self

    def add_tool(self, tool_class: str, **params: Any) -> Self:
        """Add a tool to the bot configuration.

        Args:
            tool_class: Fully qualified tool class name.
            **params: Tool constructor parameters.

        Returns:
            self for method chaining.
        """
        tools = self._config.setdefault("tools", [])
        tool_entry: dict[str, Any] = {"class": tool_class}
        if params:
            tool_entry["params"] = dict(params)
        tools.append(tool_entry)
        return self

    def add_middleware(self, middleware_class: str, **params: Any) -> Self:
        """Add middleware to the bot configuration.

        Args:
            middleware_class: Fully qualified middleware class name.
            **params: Middleware constructor parameters.

        Returns:
            self for method chaining.
        """
        middleware = self._config.setdefault("middleware", [])
        mw_entry: dict[str, Any] = {"class": middleware_class}
        if params:
            mw_entry["params"] = dict(params)
        middleware.append(mw_entry)
        return self

    # -- Extension point --

    def set_custom_section(self, key: str, value: Any) -> Self:
        """Set a custom (domain-specific) config section.

        This is the extension point for consumers to add sections like
        ``educational``, ``customer_service``, ``domain``, etc.

        Args:
            key: Section key name.
            value: Section value (dict, list, or scalar).

        Returns:
            self for method chaining.
        """
        self._custom_sections[key] = value
        return self

    # -- Template integration --

    def from_template(
        self,
        template: ConfigTemplate,
        variables: dict[str, Any],
    ) -> Self:
        """Initialize the builder from a template.

        Deep-copies the template structure, substitutes variables, and
        uses the result as the builder's base configuration.

        Args:
            template: The template to apply.
            variables: Variable values for substitution.

        Returns:
            self for method chaining.
        """
        from .templates import _build_variable_map

        from dataknobs_config.template_vars import substitute_template_vars

        var_map = _build_variable_map(template, variables)
        structure = copy.deepcopy(template.structure)
        resolved: dict[str, Any] = substitute_template_vars(
            structure, var_map, preserve_missing=True
        )

        # If structure has a 'bot' key, use its contents as the config
        if "bot" in resolved:
            self._config = dict(resolved.pop("bot"))
            # Remaining top-level keys become custom sections
            for key, value in resolved.items():
                self._custom_sections[key] = value
        else:
            self._config = resolved

        return self

    def merge_overrides(self, overrides: dict[str, Any]) -> Self:
        """Merge override values into the current configuration.

        Performs recursive dict merge for nested dictionaries.

        Args:
            overrides: Override values to merge.

        Returns:
            self for method chaining.
        """
        self._config = _deep_merge(self._config, overrides)
        return self

    # -- Output --

    def validate(self) -> ValidationResult:
        """Validate the current configuration.

        Returns:
            ValidationResult with any errors and warnings.
        """
        config = self._build_internal()
        return self._validator.validate(config)

    def build(self) -> dict[str, Any]:
        """Build the flat configuration dict.

        The returned dict is compatible with ``DynaBot.from_config()``.
        Validates before returning and raises ValueError if there are errors.

        Returns:
            Flat configuration dictionary.

        Raises:
            ValueError: If the configuration has validation errors.
        """
        config = self._build_internal()
        result = self._validator.validate(config)
        if not result.valid:
            raise ValueError(
                "Configuration validation failed:\n"
                + "\n".join(f"  - {e}" for e in result.errors)
            )
        for warning in result.warnings:
            logger.warning("Config warning: %s", warning)
        return config

    def build_portable(self) -> dict[str, Any]:
        """Build the portable configuration with $resource references.

        Wraps the config under a ``bot`` key and includes any custom
        sections as top-level siblings.

        Returns:
            Portable configuration dict with ``bot`` wrapper.

        Raises:
            ValueError: If the configuration has validation errors.
        """
        config = self._build_internal()
        result = self._validator.validate(config)
        if not result.valid:
            raise ValueError(
                "Configuration validation failed:\n"
                + "\n".join(f"  - {e}" for e in result.errors)
            )
        for warning in result.warnings:
            logger.warning("Config warning: %s", warning)

        # Separate core bot config from custom sections
        bot_config: dict[str, Any] = {}
        custom: dict[str, Any] = {}
        for key, value in config.items():
            if key in self._custom_sections:
                custom[key] = value
            else:
                bot_config[key] = value

        portable: dict[str, Any] = {"bot": bot_config}
        portable.update(custom)
        return portable

    def to_yaml(self) -> str:
        """Serialize the portable configuration as YAML.

        Returns:
            YAML string representation.
        """
        portable = self.build_portable()
        return yaml.dump(portable, default_flow_style=False, sort_keys=False)

    def reset(self) -> Self:
        """Reset the builder to an empty state.

        Returns:
            self for method chaining.
        """
        self._config = {}
        self._custom_sections = {}
        return self

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DynaBotConfigBuilder:
        """Create a builder pre-populated from an existing config.

        Supports both flat format and portable format (with ``bot`` wrapper).

        Args:
            config: Existing configuration dictionary.

        Returns:
            A new builder instance with the config loaded.
        """
        builder = cls()
        if "bot" in config:
            bot = dict(config["bot"])
            builder._config = bot
            for key, value in config.items():
                if key != "bot":
                    builder._custom_sections[key] = value
        else:
            builder._config = dict(config)
        return builder

    # -- Private --

    def _build_internal(self) -> dict[str, Any]:
        """Build the internal config dict with custom sections merged."""
        config = dict(self._config)
        config.update(self._custom_sections)
        return config


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overrides into base dict.

    Args:
        base: Base dictionary.
        overrides: Override values to merge in.

    Returns:
        New merged dictionary.
    """
    result = dict(base)
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
