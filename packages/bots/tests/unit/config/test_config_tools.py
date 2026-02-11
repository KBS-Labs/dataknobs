"""Tests for tools/config_tools.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from dataknobs_llm.tools.context import ToolExecutionContext, WizardStateSnapshot

from dataknobs_bots.config.builder import DynaBotConfigBuilder
from dataknobs_bots.config.drafts import ConfigDraftManager
from dataknobs_bots.config.templates import (
    ConfigTemplate,
    ConfigTemplateRegistry,
    TemplateVariable,
)
from dataknobs_bots.config.validation import ConfigValidator
from dataknobs_bots.tools.config_tools import (
    GetTemplateDetailsTool,
    ListAvailableToolsTool,
    ListTemplatesTool,
    PreviewConfigTool,
    SaveConfigTool,
    ValidateConfigTool,
)


def _make_context(
    wizard_data: dict[str, Any] | None = None,
) -> ToolExecutionContext:
    """Create a ToolExecutionContext with wizard state."""
    if wizard_data is not None:
        wizard_state = WizardStateSnapshot(
            current_stage="test",
            collected_data=wizard_data,
            history=["test"],
            completed=False,
        )
    else:
        wizard_state = None
    return ToolExecutionContext(
        conversation_id="test-conv",
        user_id="test-user",
        wizard_state=wizard_state,
    )


def _make_registry() -> ConfigTemplateRegistry:
    """Create a registry with test templates."""
    registry = ConfigTemplateRegistry()
    registry.register(
        ConfigTemplate(
            name="basic",
            description="Basic bot",
            version="1.0.0",
            tags=["simple"],
            variables=[
                TemplateVariable(name="bot_name", required=True),
            ],
            structure={
                "bot": {
                    "llm": {"provider": "ollama"},
                    "conversation_storage": {"backend": "memory"},
                    "system_prompt": "I am {{bot_name}}",
                }
            },
        )
    )
    registry.register(
        ConfigTemplate(
            name="advanced",
            description="Advanced bot",
            tags=["advanced", "rag"],
            variables=[
                TemplateVariable(name="bot_name", required=True),
                TemplateVariable(name="subject", required=True),
            ],
            structure={
                "bot": {
                    "llm": {"provider": "ollama"},
                    "conversation_storage": {"backend": "memory"},
                    "knowledge_base": {"enabled": True},
                }
            },
        )
    )
    return registry


def _basic_builder_factory(wizard_data: dict[str, Any]) -> DynaBotConfigBuilder:
    """Simple builder factory for testing."""
    builder = (
        DynaBotConfigBuilder()
        .set_llm(
            wizard_data.get("llm_provider", "ollama"),
            model=wizard_data.get("llm_model", "llama3.2"),
        )
        .set_conversation_storage(
            wizard_data.get("storage_backend", "memory")
        )
    )
    if "system_prompt" in wizard_data:
        builder.set_system_prompt(content=wizard_data["system_prompt"])
    return builder


def _portable_builder_factory(wizard_data: dict[str, Any]) -> DynaBotConfigBuilder:
    """Builder factory that adds custom sections (for portable test)."""
    builder = (
        DynaBotConfigBuilder()
        .set_llm(
            wizard_data.get("llm_provider", "ollama"),
            model=wizard_data.get("llm_model", "llama3.2"),
        )
        .set_conversation_storage(
            wizard_data.get("storage_backend", "memory")
        )
    )
    if wizard_data.get("domain_id"):
        builder.set_custom_section("domain", {
            "id": wizard_data["domain_id"],
        })
    return builder


class TestListTemplatesTool:
    """Tests for ListTemplatesTool."""

    @pytest.mark.asyncio
    async def test_list_all(self) -> None:
        tool = ListTemplatesTool(template_registry=_make_registry())
        result = await tool.execute_with_context(_make_context())
        assert result["count"] == 2
        names = [t["name"] for t in result["templates"]]
        assert "basic" in names
        assert "advanced" in names

    @pytest.mark.asyncio
    async def test_list_with_tags(self) -> None:
        tool = ListTemplatesTool(template_registry=_make_registry())
        result = await tool.execute_with_context(
            _make_context(), tags=["rag"]
        )
        assert result["count"] == 1
        assert result["templates"][0]["name"] == "advanced"

    @pytest.mark.asyncio
    async def test_list_empty_tag_match(self) -> None:
        tool = ListTemplatesTool(template_registry=_make_registry())
        result = await tool.execute_with_context(
            _make_context(), tags=["nonexistent"]
        )
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_schema(self) -> None:
        tool = ListTemplatesTool(template_registry=_make_registry())
        assert tool.schema["type"] == "object"
        assert "tags" in tool.schema["properties"]


class TestGetTemplateDetailsTool:
    """Tests for GetTemplateDetailsTool."""

    @pytest.mark.asyncio
    async def test_get_existing(self) -> None:
        tool = GetTemplateDetailsTool(template_registry=_make_registry())
        result = await tool.execute_with_context(
            _make_context(), template_name="basic"
        )
        assert result["name"] == "basic"
        assert result["description"] == "Basic bot"
        assert len(result["variables"]) == 1
        assert len(result["required_variables"]) == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent(self) -> None:
        tool = GetTemplateDetailsTool(template_registry=_make_registry())
        result = await tool.execute_with_context(
            _make_context(), template_name="missing"
        )
        assert "error" in result
        assert "available" in result

    @pytest.mark.asyncio
    async def test_schema(self) -> None:
        tool = GetTemplateDetailsTool(template_registry=_make_registry())
        assert "template_name" in tool.schema["properties"]
        assert "template_name" in tool.schema["required"]


class TestPreviewConfigTool:
    """Tests for PreviewConfigTool."""

    @pytest.mark.asyncio
    async def test_preview_summary(self) -> None:
        tool = PreviewConfigTool(builder_factory=_basic_builder_factory)
        context = _make_context(
            {"llm_provider": "ollama", "llm_model": "llama3.2"}
        )
        result = await tool.execute_with_context(context, format="summary")
        assert "sections" in result

    @pytest.mark.asyncio
    async def test_preview_full(self) -> None:
        tool = PreviewConfigTool(builder_factory=_basic_builder_factory)
        context = _make_context(
            {"llm_provider": "ollama", "storage_backend": "memory"}
        )
        result = await tool.execute_with_context(context, format="full")
        assert "config" in result
        assert result["config"]["llm"]["provider"] == "ollama"

    @pytest.mark.asyncio
    async def test_preview_yaml(self) -> None:
        tool = PreviewConfigTool(builder_factory=_basic_builder_factory)
        context = _make_context(
            {"llm_provider": "openai", "storage_backend": "sqlite"}
        )
        result = await tool.execute_with_context(context, format="yaml")
        assert "yaml" in result
        parsed = yaml.safe_load(result["yaml"])
        assert parsed["llm"]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_preview_no_wizard_data(self) -> None:
        tool = PreviewConfigTool(builder_factory=_basic_builder_factory)
        result = await tool.execute_with_context(_make_context())
        assert "error" in result

    @pytest.mark.asyncio
    async def test_preview_builder_error(self) -> None:
        def bad_factory(data: dict[str, Any]) -> DynaBotConfigBuilder:
            raise ValueError("oops")

        tool = PreviewConfigTool(builder_factory=bad_factory)
        context = _make_context({"some": "data"})
        result = await tool.execute_with_context(context)
        assert "error" in result


class TestValidateConfigTool:
    """Tests for ValidateConfigTool."""

    @pytest.mark.asyncio
    async def test_valid_config(self) -> None:
        validator = ConfigValidator()
        tool = ValidateConfigTool(
            validator=validator, builder_factory=_basic_builder_factory
        )
        context = _make_context(
            {"llm_provider": "ollama", "storage_backend": "memory"}
        )
        result = await tool.execute_with_context(context)
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_no_wizard_data(self) -> None:
        validator = ConfigValidator()
        tool = ValidateConfigTool(validator=validator)
        result = await tool.execute_with_context(_make_context())
        assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_without_builder_factory(self) -> None:
        validator = ConfigValidator()
        tool = ValidateConfigTool(validator=validator)
        context = _make_context(
            {
                "llm": {"provider": "ollama"},
                "conversation_storage": {"backend": "memory"},
            }
        )
        result = await tool.execute_with_context(context)
        assert result["valid"] is True


class TestSaveConfigTool:
    """Tests for SaveConfigTool."""

    @pytest.mark.asyncio
    async def test_save_basic(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        tool = SaveConfigTool(
            draft_manager=manager,
            builder_factory=_basic_builder_factory,
        )
        context = _make_context(
            {
                "domain_id": "test-bot",
                "llm_provider": "ollama",
                "storage_backend": "memory",
            }
        )
        result = await tool.execute_with_context(context)
        assert result["success"] is True
        assert result["config_name"] == "test-bot"
        assert (tmp_path / "test-bot.yaml").exists()

    @pytest.mark.asyncio
    async def test_save_with_draft(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        draft_id = manager.create_draft(
            {"llm": {"provider": "ollama"}, "conversation_storage": {"backend": "memory"}}
        )

        tool = SaveConfigTool(
            draft_manager=manager,
            builder_factory=_basic_builder_factory,
        )
        context = _make_context(
            {
                "_draft_id": draft_id,
                "domain_id": "test-bot",
                "llm_provider": "ollama",
                "storage_backend": "memory",
            }
        )
        result = await tool.execute_with_context(context)
        assert result["success"] is True
        # Draft file should be cleaned up
        assert not (tmp_path / f"_draft-{draft_id}.yaml").exists()

    @pytest.mark.asyncio
    async def test_save_with_callback(self, tmp_path: Path) -> None:
        saved_args: list[tuple[str, dict[str, Any]]] = []

        def on_save(name: str, config: dict[str, Any]) -> None:
            saved_args.append((name, config))

        manager = ConfigDraftManager(output_dir=tmp_path)
        tool = SaveConfigTool(
            draft_manager=manager,
            on_save=on_save,
            builder_factory=_basic_builder_factory,
        )
        context = _make_context(
            {
                "domain_id": "cb-bot",
                "llm_provider": "ollama",
                "storage_backend": "memory",
            }
        )
        result = await tool.execute_with_context(context)
        assert result["success"] is True
        assert len(saved_args) == 1
        assert saved_args[0][0] == "cb-bot"

    @pytest.mark.asyncio
    async def test_save_no_wizard_data(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        tool = SaveConfigTool(draft_manager=manager)
        result = await tool.execute_with_context(_make_context())
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_save_no_name(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        tool = SaveConfigTool(
            draft_manager=manager,
            builder_factory=_basic_builder_factory,
        )
        context = _make_context(
            {"llm_provider": "ollama", "storage_backend": "memory"}
        )
        result = await tool.execute_with_context(context)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_save_with_explicit_name(self, tmp_path: Path) -> None:
        manager = ConfigDraftManager(output_dir=tmp_path)
        tool = SaveConfigTool(
            draft_manager=manager,
            builder_factory=_basic_builder_factory,
        )
        context = _make_context(
            {"llm_provider": "ollama", "storage_backend": "memory"}
        )
        result = await tool.execute_with_context(
            context, config_name="explicit-name"
        )
        assert result["success"] is True
        assert result["config_name"] == "explicit-name"
        assert (tmp_path / "explicit-name.yaml").exists()

    @pytest.mark.asyncio
    async def test_save_portable(self, tmp_path: Path) -> None:
        """Verify portable=True uses build_portable() (bot wrapper)."""
        manager = ConfigDraftManager(output_dir=tmp_path)
        tool = SaveConfigTool(
            draft_manager=manager,
            builder_factory=_portable_builder_factory,
            portable=True,
        )
        context = _make_context(
            {
                "domain_id": "portable-bot",
                "llm_provider": "ollama",
                "storage_backend": "memory",
            }
        )
        result = await tool.execute_with_context(context)
        assert result["success"] is True

        # Read back and verify portable format (bot wrapper + custom section)
        saved = yaml.safe_load((tmp_path / "portable-bot.yaml").read_text())
        assert "bot" in saved, "Portable format should have 'bot' wrapper key"
        assert saved["bot"]["llm"]["provider"] == "ollama"
        assert saved["domain"]["id"] == "portable-bot"

    @pytest.mark.asyncio
    async def test_save_non_portable(self, tmp_path: Path) -> None:
        """Verify portable=False (default) uses _build_internal() (flat)."""
        manager = ConfigDraftManager(output_dir=tmp_path)
        tool = SaveConfigTool(
            draft_manager=manager,
            builder_factory=_portable_builder_factory,
        )
        context = _make_context(
            {
                "domain_id": "flat-bot",
                "llm_provider": "ollama",
                "storage_backend": "memory",
            }
        )
        result = await tool.execute_with_context(context)
        assert result["success"] is True

        saved = yaml.safe_load((tmp_path / "flat-bot.yaml").read_text())
        assert "bot" not in saved, "Flat format should NOT have 'bot' wrapper"
        assert saved["llm"]["provider"] == "ollama"


class TestListAvailableToolsTool:
    """Tests for ListAvailableToolsTool."""

    @pytest.mark.asyncio
    async def test_list_all(self) -> None:
        catalog = [
            {"name": "search", "description": "Search", "category": "info"},
            {"name": "calc", "description": "Calculator", "category": "math"},
            {"name": "weather", "description": "Weather", "category": "info"},
        ]
        tool = ListAvailableToolsTool(available_tools=catalog)
        result = await tool.execute_with_context(_make_context())
        assert result["count"] == 3
        assert len(result["tools"]) == 3
        assert set(result["categories"]) == {"info", "math"}

    @pytest.mark.asyncio
    async def test_filter_by_category(self) -> None:
        catalog = [
            {"name": "search", "description": "Search", "category": "info"},
            {"name": "calc", "description": "Calculator", "category": "math"},
            {"name": "weather", "description": "Weather", "category": "info"},
        ]
        tool = ListAvailableToolsTool(available_tools=catalog)
        result = await tool.execute_with_context(
            _make_context(), category="info"
        )
        assert result["count"] == 2
        names = [t["name"] for t in result["tools"]]
        assert "search" in names
        assert "weather" in names
        # Categories still lists all available categories
        assert set(result["categories"]) == {"info", "math"}

    @pytest.mark.asyncio
    async def test_filter_case_insensitive(self) -> None:
        catalog = [
            {"name": "search", "description": "Search", "category": "Info"},
        ]
        tool = ListAvailableToolsTool(available_tools=catalog)
        result = await tool.execute_with_context(
            _make_context(), category="info"
        )
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_empty_catalog(self) -> None:
        tool = ListAvailableToolsTool(available_tools=[])
        result = await tool.execute_with_context(_make_context())
        assert result["count"] == 0
        assert result["tools"] == []
        assert result["categories"] == []

    @pytest.mark.asyncio
    async def test_schema(self) -> None:
        tool = ListAvailableToolsTool(available_tools=[])
        assert tool.schema["type"] == "object"
        assert "category" in tool.schema["properties"]
