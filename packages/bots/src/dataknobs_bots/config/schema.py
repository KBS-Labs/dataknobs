"""DynaBot configuration schema registry.

Provides a queryable registry of valid DynaBot config options. A ConfigBot
needs to ask questions like "what LLM providers are available?" or "what does
memory config look like?" -- this module answers those questions.

Example:
    ```python
    from dataknobs_bots.config.schema import DynaBotConfigSchema

    schema = DynaBotConfigSchema()

    # Query available LLM providers
    providers = schema.get_valid_options("llm", "provider")
    # ["ollama", "openai", "anthropic", "huggingface", "echo"]

    # Register consumer-specific extension
    schema.register_extension("educational", {
        "type": "object",
        "properties": {
            "enable_progress_tracking": {"type": "boolean"},
        }
    })

    # Generate LLM-friendly description for system prompts
    description = schema.to_description()
    ```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .validation import ValidationResult, _validate_against_schema

logger = logging.getLogger(__name__)


@dataclass
class ComponentSchema:
    """Schema for a single DynaBot config component.

    Attributes:
        name: Component name (e.g., 'llm', 'memory').
        description: Human-readable description.
        schema: JSON Schema-like definition of valid options.
        required: Whether this component is required in a valid config.
    """

    name: str
    description: str
    schema: dict[str, Any] = field(default_factory=dict)
    required: bool = False

    def get_valid_options(self, field_name: str) -> list[str]:
        """Get valid options for a field within this component.

        Args:
            field_name: The field name to query.

        Returns:
            List of valid option strings, or empty list if unconstrained.
        """
        properties = self.schema.get("properties", {})
        prop = properties.get(field_name, {})
        enum_values: list[str] = prop.get("enum", [])
        return enum_values


class DynaBotConfigSchema:
    """Queryable registry of valid DynaBot configuration options.

    Auto-registers the 8 default DynaBot components on initialization.
    Consumers can register additional extensions for domain-specific sections.
    """

    def __init__(self) -> None:
        self._components: dict[str, ComponentSchema] = {}
        self._extensions: dict[str, ComponentSchema] = {}
        self._register_defaults()

    def register_component(
        self,
        name: str,
        schema: dict[str, Any],
        description: str = "",
        required: bool = False,
    ) -> None:
        """Register a core DynaBot component schema.

        Args:
            name: Component name.
            schema: JSON Schema-like definition.
            description: Human-readable description.
            required: Whether this component is required.
        """
        self._components[name] = ComponentSchema(
            name=name,
            description=description,
            schema=schema,
            required=required,
        )
        logger.debug("Registered component schema: %s", name)

    def register_extension(
        self,
        name: str,
        schema: dict[str, Any],
        description: str = "",
    ) -> None:
        """Register a consumer-specific config extension.

        Extensions are domain-specific sections (e.g., 'educational',
        'customer_service') that aren't part of the core DynaBot schema.

        Args:
            name: Extension name.
            schema: JSON Schema-like definition.
            description: Human-readable description.
        """
        self._extensions[name] = ComponentSchema(
            name=name,
            description=description or f"Extension: {name}",
            schema=schema,
        )
        logger.debug("Registered extension schema: %s", name)

    def get_component_schema(self, name: str) -> dict[str, Any] | None:
        """Get the JSON Schema for a component.

        Args:
            name: Component name.

        Returns:
            JSON Schema dict, or None if not registered.
        """
        component = self._components.get(name)
        if component is not None:
            return component.schema
        return None

    def get_extension_schema(self, name: str) -> dict[str, Any] | None:
        """Get the JSON Schema for an extension.

        Args:
            name: Extension name.

        Returns:
            JSON Schema dict, or None if not registered.
        """
        ext = self._extensions.get(name)
        if ext is not None:
            return ext.schema
        return None

    def get_valid_options(self, component: str, field_name: str) -> list[str]:
        """Get valid options for a field within a component or extension.

        Args:
            component: Component or extension name.
            field_name: Field name to query.

        Returns:
            List of valid option strings.
        """
        comp = self._components.get(component) or self._extensions.get(component)
        if comp is not None:
            return comp.get_valid_options(field_name)
        return []

    def validate(self, config: dict[str, Any]) -> ValidationResult:
        """Validate a config against all registered schemas.

        Args:
            config: Full DynaBot configuration dict.

        Returns:
            ValidationResult with all schema violations.
        """
        result = ValidationResult.ok()
        bot = config.get("bot", config)

        for name, comp in self._components.items():
            if comp.required and name not in bot:
                result = result.merge(
                    ValidationResult.error(f"Missing required component: {name}")
                )
            if name in bot and isinstance(bot[name], dict):
                result = result.merge(
                    _validate_against_schema(name, bot[name], comp.schema)
                )

        for name, ext in self._extensions.items():
            if name in bot and isinstance(bot[name], dict):
                result = result.merge(
                    _validate_against_schema(name, bot[name], ext.schema)
                )

        return result

    def get_full_schema(self) -> dict[str, Any]:
        """Get the combined schema for all components and extensions.

        Returns:
            Dict mapping component/extension names to their schemas.
        """
        result: dict[str, Any] = {}
        for name, comp in self._components.items():
            result[name] = {
                "description": comp.description,
                "required": comp.required,
                "schema": comp.schema,
            }
        for name, ext in self._extensions.items():
            result[name] = {
                "description": ext.description,
                "required": False,
                "extension": True,
                "schema": ext.schema,
            }
        return result

    def to_description(self) -> str:
        """Generate a human-readable description for LLM system prompts.

        Returns:
            Structured text describing all available configuration options.
        """
        lines: list[str] = ["# DynaBot Configuration Options", ""]

        lines.append("## Core Components")
        lines.append("")
        for name, comp in self._components.items():
            req = " (required)" if comp.required else " (optional)"
            lines.append(f"### {name}{req}")
            if comp.description:
                lines.append(comp.description)
            props = comp.schema.get("properties", {})
            if props:
                lines.append("")
                for field_name, field_schema in props.items():
                    desc = field_schema.get("description", "")
                    enum_values = field_schema.get("enum")
                    line = f"- **{field_name}**"
                    if desc:
                        line += f": {desc}"
                    if enum_values:
                        line += f" (options: {', '.join(str(v) for v in enum_values)})"
                    lines.append(line)
            lines.append("")

        if self._extensions:
            lines.append("## Extensions")
            lines.append("")
            for name, ext in self._extensions.items():
                lines.append(f"### {name}")
                if ext.description:
                    lines.append(ext.description)
                props = ext.schema.get("properties", {})
                if props:
                    lines.append("")
                    for field_name, field_schema in props.items():
                        desc = field_schema.get("description", "")
                        line = f"- **{field_name}**"
                        if desc:
                            line += f": {desc}"
                        lines.append(line)
                lines.append("")

        return "\n".join(lines)

    def _register_defaults(self) -> None:
        """Register the 8 default DynaBot config components."""
        self.register_component(
            "llm",
            description="LLM provider configuration.",
            schema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "LLM provider to use",
                        "enum": [
                            "ollama",
                            "openai",
                            "anthropic",
                            "huggingface",
                            "echo",
                        ],
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name or identifier",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature (0.0-2.0)",
                        "minimum": 0.0,
                        "maximum": 2.0,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens in response",
                        "minimum": 1,
                    },
                },
            },
            required=True,
        )

        self.register_component(
            "conversation_storage",
            description="Backend for storing conversation history.",
            schema={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "description": "Storage backend type",
                        "enum": [
                            "memory",
                            "sqlite",
                            "postgres",
                            "elasticsearch",
                            "s3",
                            "duckdb",
                            "file",
                        ],
                    },
                },
            },
            required=True,
        )

        self.register_component(
            "memory",
            description="Conversation memory configuration.",
            schema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Memory type",
                        "enum": ["buffer", "vector"],
                    },
                    "max_messages": {
                        "type": "integer",
                        "description": "Maximum messages to retain",
                        "minimum": 1,
                    },
                },
            },
        )

        self.register_component(
            "reasoning",
            description="Reasoning strategy for the bot.",
            schema={
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "description": "Reasoning strategy",
                        "enum": ["simple", "react", "wizard"],
                    },
                },
            },
        )

        self.register_component(
            "knowledge_base",
            description="RAG knowledge base configuration.",
            schema={
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "Whether knowledge base is enabled",
                    },
                    "type": {
                        "type": "string",
                        "description": "Knowledge base type",
                        "enum": ["rag"],
                    },
                    "vector_store": {
                        "type": "object",
                        "description": "Vector store configuration",
                    },
                    "embedding": {
                        "type": "object",
                        "description": "Embedding provider configuration",
                    },
                },
            },
        )

        self.register_component(
            "tools",
            description="LLM-callable tools available to the bot.",
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "class": {
                            "type": "string",
                            "description": "Fully qualified tool class name",
                        },
                        "params": {
                            "type": "object",
                            "description": "Tool constructor parameters",
                        },
                    },
                },
            },
        )

        self.register_component(
            "middleware",
            description="Request/response middleware pipeline.",
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "class": {
                            "type": "string",
                            "description": "Fully qualified middleware class name",
                        },
                        "params": {
                            "type": "object",
                            "description": "Middleware constructor parameters",
                        },
                    },
                },
            },
        )

        self.register_component(
            "system_prompt",
            description="System prompt configuration.",
            schema={
                "type": ["string", "object"],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Prompt template name",
                    },
                    "content": {
                        "type": "string",
                        "description": "Inline prompt content",
                    },
                    "rag_configs": {
                        "type": "array",
                        "description": "RAG configurations for prompt enhancement",
                    },
                },
            },
        )
