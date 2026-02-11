"""Tests for config/schema.py."""

from __future__ import annotations

from typing import Any

from dataknobs_bots.config.schema import ComponentSchema, DynaBotConfigSchema


class TestComponentSchema:
    """Tests for ComponentSchema dataclass."""

    def test_get_valid_options_with_enum(self) -> None:
        schema = ComponentSchema(
            name="llm",
            description="LLM config",
            schema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "enum": ["ollama", "openai"],
                    },
                },
            },
        )
        assert schema.get_valid_options("provider") == ["ollama", "openai"]

    def test_get_valid_options_no_enum(self) -> None:
        schema = ComponentSchema(
            name="llm",
            description="LLM config",
            schema={
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                },
            },
        )
        assert schema.get_valid_options("model") == []

    def test_get_valid_options_missing_field(self) -> None:
        schema = ComponentSchema(name="test", description="", schema={})
        assert schema.get_valid_options("nonexistent") == []


class TestDynaBotConfigSchema:
    """Tests for DynaBotConfigSchema."""

    def test_default_components_registered(self) -> None:
        schema = DynaBotConfigSchema()
        # Should have 8 default components
        assert schema.get_component_schema("llm") is not None
        assert schema.get_component_schema("conversation_storage") is not None
        assert schema.get_component_schema("memory") is not None
        assert schema.get_component_schema("reasoning") is not None
        assert schema.get_component_schema("knowledge_base") is not None
        assert schema.get_component_schema("tools") is not None
        assert schema.get_component_schema("middleware") is not None
        assert schema.get_component_schema("system_prompt") is not None

    def test_get_valid_options_llm_provider(self) -> None:
        schema = DynaBotConfigSchema()
        providers = schema.get_valid_options("llm", "provider")
        assert "ollama" in providers
        assert "openai" in providers
        assert "anthropic" in providers

    def test_get_valid_options_storage_backend(self) -> None:
        schema = DynaBotConfigSchema()
        backends = schema.get_valid_options("conversation_storage", "backend")
        assert "memory" in backends
        assert "sqlite" in backends
        assert "postgres" in backends

    def test_get_valid_options_memory_type(self) -> None:
        schema = DynaBotConfigSchema()
        types = schema.get_valid_options("memory", "type")
        assert "buffer" in types
        assert "vector" in types

    def test_get_valid_options_reasoning_strategy(self) -> None:
        schema = DynaBotConfigSchema()
        strategies = schema.get_valid_options("reasoning", "strategy")
        assert "simple" in strategies
        assert "react" in strategies
        assert "wizard" in strategies

    def test_register_extension(self) -> None:
        schema = DynaBotConfigSchema()
        schema.register_extension(
            "educational",
            {
                "type": "object",
                "properties": {
                    "assessment_mode": {
                        "type": "string",
                        "enum": ["formative", "summative"],
                    },
                },
            },
        )
        ext_schema = schema.get_extension_schema("educational")
        assert ext_schema is not None
        modes = schema.get_valid_options("educational", "assessment_mode")
        assert modes == ["formative", "summative"]

    def test_get_nonexistent_schema(self) -> None:
        schema = DynaBotConfigSchema()
        assert schema.get_component_schema("nonexistent") is None
        assert schema.get_extension_schema("nonexistent") is None

    def test_get_valid_options_nonexistent_component(self) -> None:
        schema = DynaBotConfigSchema()
        assert schema.get_valid_options("nonexistent", "field") == []

    def test_validate_valid_config(self) -> None:
        schema = DynaBotConfigSchema()
        config: dict[str, Any] = {
            "llm": {"provider": "ollama", "model": "llama3.2"},
            "conversation_storage": {"backend": "memory"},
        }
        result = schema.validate(config)
        assert result.valid is True

    def test_validate_invalid_provider(self) -> None:
        schema = DynaBotConfigSchema()
        config: dict[str, Any] = {
            "llm": {"provider": "nonexistent"},
            "conversation_storage": {"backend": "memory"},
        }
        result = schema.validate(config)
        assert result.valid is False

    def test_validate_missing_required(self) -> None:
        schema = DynaBotConfigSchema()
        config: dict[str, Any] = {
            "conversation_storage": {"backend": "memory"},
        }
        result = schema.validate(config)
        assert result.valid is False
        assert any("llm" in e for e in result.errors)

    def test_validate_extension_section(self) -> None:
        schema = DynaBotConfigSchema()
        schema.register_extension(
            "educational",
            {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["quiz", "tutor"]},
                },
            },
        )
        config: dict[str, Any] = {
            "llm": {"provider": "ollama"},
            "conversation_storage": {"backend": "memory"},
            "educational": {"mode": "invalid"},
        }
        result = schema.validate(config)
        assert result.valid is False
        assert any("invalid" in e for e in result.errors)

    def test_validate_resource_refs_skipped(self) -> None:
        schema = DynaBotConfigSchema()
        config: dict[str, Any] = {
            "llm": {"$resource": "default", "type": "llm_providers"},
            "conversation_storage": {"$resource": "db", "type": "databases"},
        }
        result = schema.validate(config)
        assert result.valid is True

    def test_get_full_schema(self) -> None:
        schema = DynaBotConfigSchema()
        full = schema.get_full_schema()
        assert "llm" in full
        assert full["llm"]["required"] is True
        assert "memory" in full
        assert full["memory"]["required"] is False

    def test_get_full_schema_with_extensions(self) -> None:
        schema = DynaBotConfigSchema()
        schema.register_extension("custom", {"type": "object", "properties": {}})
        full = schema.get_full_schema()
        assert "custom" in full
        assert full["custom"].get("extension") is True

    def test_to_description(self) -> None:
        schema = DynaBotConfigSchema()
        desc = schema.to_description()
        assert "# DynaBot Configuration Options" in desc
        assert "## Core Components" in desc
        assert "### llm" in desc
        assert "ollama" in desc
        assert "openai" in desc

    def test_to_description_with_extensions(self) -> None:
        schema = DynaBotConfigSchema()
        schema.register_extension(
            "educational",
            {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "description": "Learning mode"},
                },
            },
            description="Educational settings",
        )
        desc = schema.to_description()
        assert "## Extensions" in desc
        assert "### educational" in desc
        assert "Learning mode" in desc
