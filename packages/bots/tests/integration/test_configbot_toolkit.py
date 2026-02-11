"""Integration test for the ConfigBot toolkit end-to-end flow.

Tests the full lifecycle: template -> builder -> validate -> draft -> finalize.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from dataknobs_bots.config.builder import DynaBotConfigBuilder
from dataknobs_bots.config.drafts import ConfigDraftManager
from dataknobs_bots.config.schema import DynaBotConfigSchema
from dataknobs_bots.config.templates import (
    ConfigTemplate,
    ConfigTemplateRegistry,
    TemplateVariable,
)
from dataknobs_bots.config.validation import ConfigValidator, ValidationResult


class TestConfigToolkitEndToEnd:
    """End-to-end integration tests for the ConfigBot toolkit."""

    def test_template_to_builder_to_validate_to_draft_to_finalize(
        self, tmp_path: Path
    ) -> None:
        """Full lifecycle: template -> builder -> validate -> draft -> finalize."""
        # 1. Set up schema with an extension
        schema = DynaBotConfigSchema()
        schema.register_extension(
            "domain",
            {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                },
            },
        )

        # 2. Register a template
        registry = ConfigTemplateRegistry()
        template = ConfigTemplate(
            name="test_bot",
            description="A test bot template",
            version="1.0.0",
            tags=["test"],
            variables=[
                TemplateVariable(name="bot_name", required=True),
                TemplateVariable(name="domain_id", required=True),
                TemplateVariable(name="temperature", default=0.7),
            ],
            structure={
                "bot": {
                    "llm": {
                        "$resource": "default",
                        "type": "llm_providers",
                        "temperature": "{{temperature}}",
                    },
                    "conversation_storage": {
                        "$resource": "conversations",
                        "type": "databases",
                    },
                    "system_prompt": "I am {{bot_name}}, here to help.",
                    "memory": {"type": "buffer", "max_messages": 50},
                },
                "domain": {
                    "id": "{{domain_id}}",
                    "name": "{{bot_name}}",
                },
            },
        )
        registry.register(template)

        # 3. Validate template variables
        vars_result = registry.validate_variables(
            "test_bot", {"bot_name": "Helper", "domain_id": "helper-bot"}
        )
        assert vars_result.valid is True

        # 4. Build config from template
        builder = DynaBotConfigBuilder(schema=schema).from_template(
            template,
            {"bot_name": "Helper", "domain_id": "helper-bot"},
        )
        result = builder.validate()
        assert result.valid is True

        # 5. Get the flat config
        config = builder.build()
        assert config["llm"]["$resource"] == "default"
        assert config["system_prompt"] == "I am Helper, here to help."

        # 6. Get portable config
        portable = builder.build_portable()
        assert "bot" in portable
        assert "domain" in portable
        assert portable["domain"]["id"] == "helper-bot"

        # 7. Validate with ConfigValidator
        validator = ConfigValidator(schema=schema)
        val_result = validator.validate(config)
        assert val_result.valid is True

        # 8. Draft lifecycle
        draft_mgr = ConfigDraftManager(output_dir=tmp_path)
        draft_id = draft_mgr.create_draft(portable, stage="configure_llm")

        # Update draft
        draft_mgr.update_draft(
            draft_id, portable, stage="review", config_name="helper-bot"
        )

        # Verify draft exists
        draft_result = draft_mgr.get_draft(draft_id)
        assert draft_result is not None
        draft_config, draft_meta = draft_result
        assert draft_meta.stage == "review"
        assert draft_meta.config_name == "helper-bot"

        # 9. Finalize
        final = draft_mgr.finalize(draft_id, final_name="helper-bot")
        assert "_draft" not in final
        assert (tmp_path / "helper-bot.yaml").exists()

        # 10. Verify saved file
        with open(tmp_path / "helper-bot.yaml") as f:
            saved = yaml.safe_load(f)
        assert saved["bot"]["system_prompt"] == "I am Helper, here to help."
        assert saved["domain"]["id"] == "helper-bot"

    def test_from_scratch_builder_flow(self) -> None:
        """Build a config from scratch without templates."""
        config = (
            DynaBotConfigBuilder()
            .set_llm("ollama", model="llama3.2", temperature=0.5)
            .set_conversation_storage("memory")
            .set_system_prompt(content="You are a coding assistant.")
            .set_memory("buffer", max_messages=100)
            .set_reasoning("react", max_iterations=5)
            .add_tool("dataknobs_bots.tools.KnowledgeSearchTool")
            .set_custom_section("domain", {"id": "code-bot", "name": "Code Bot"})
            .build()
        )

        assert config["llm"]["provider"] == "ollama"
        assert config["llm"]["model"] == "llama3.2"
        assert config["system_prompt"] == "You are a coding assistant."
        assert config["reasoning"]["strategy"] == "react"
        assert len(config["tools"]) == 1
        assert config["domain"]["id"] == "code-bot"

    def test_builder_round_trip(self) -> None:
        """Build -> build_portable -> from_config -> build should produce same config."""
        builder1 = (
            DynaBotConfigBuilder()
            .set_llm("ollama", model="llama3.2")
            .set_conversation_storage("memory")
            .set_custom_section("meta", {"version": "1.0"})
        )
        portable = builder1.build_portable()

        builder2 = DynaBotConfigBuilder.from_config(portable)
        portable2 = builder2.build_portable()

        assert portable == portable2

    def test_template_override_flow(self) -> None:
        """Apply template then override specific settings."""
        template = ConfigTemplate(
            name="base",
            variables=[
                TemplateVariable(name="bot_name", required=True),
            ],
            structure={
                "bot": {
                    "llm": {
                        "$resource": "default",
                        "type": "llm_providers",
                        "temperature": "0.7",
                    },
                    "conversation_storage": {
                        "$resource": "conversations",
                        "type": "databases",
                    },
                    "system_prompt": "I am {{bot_name}}",
                }
            },
        )

        config = (
            DynaBotConfigBuilder()
            .from_template(template, {"bot_name": "TestBot"})
            .merge_overrides({"llm": {"temperature": 0.3}})
            .build()
        )

        assert config["system_prompt"] == "I am TestBot"
        assert config["llm"]["temperature"] == 0.3
        assert config["llm"]["$resource"] == "default"

    def test_built_in_templates_loadable(self) -> None:
        """Verify built-in templates can be loaded from the package."""
        templates_dir = (
            Path(__file__).parent.parent.parent
            / "src"
            / "dataknobs_bots"
            / "config"
            / "templates"
        )
        if not templates_dir.exists():
            pytest.skip("Built-in templates directory not found")

        registry = ConfigTemplateRegistry()
        count = registry.load_from_directory(templates_dir)
        assert count >= 3

        # Verify each built-in template
        basic = registry.get("basic_assistant")
        assert basic is not None
        assert len(basic.variables) > 0

        rag = registry.get("rag_assistant")
        assert rag is not None
        assert "rag" in rag.tags

        tool = registry.get("tool_user")
        assert tool is not None
        assert "tools" in tool.tags

    def test_validation_catches_invalid_provider(self) -> None:
        """Validation should catch invalid enum values."""
        schema = DynaBotConfigSchema()
        validator = ConfigValidator(schema=schema)

        config: dict[str, Any] = {
            "llm": {"provider": "nonexistent_provider"},
            "conversation_storage": {"backend": "memory"},
        }
        result = validator.validate(config)
        assert result.valid is False
        assert any("nonexistent_provider" in e for e in result.errors)

    def test_custom_validator_integration(self) -> None:
        """Custom validators work alongside built-in validation."""
        validator = ConfigValidator()

        def check_domain_id(config: dict[str, Any]) -> ValidationResult:
            domain = config.get("domain", {})
            if isinstance(domain, dict) and "id" in domain:
                domain_id = domain["id"]
                if not isinstance(domain_id, str) or " " in domain_id:
                    return ValidationResult.error(
                        "domain.id must not contain spaces"
                    )
            return ValidationResult.ok()

        validator.register_validator("domain_id", check_domain_id)

        config: dict[str, Any] = {
            "llm": {"provider": "ollama"},
            "conversation_storage": {"backend": "memory"},
            "domain": {"id": "has spaces"},
        }
        result = validator.validate(config)
        assert result.valid is False
        assert any("spaces" in e for e in result.errors)
