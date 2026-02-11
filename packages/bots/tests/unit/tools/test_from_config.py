"""Tests for from_config() protocol and resolve_callable() utility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from dataknobs_bots.tools.resolve import resolve_callable


class TestResolveCallable:
    """Tests for resolve_callable()."""

    def test_colon_separator(self) -> None:
        result = resolve_callable(
            "dataknobs_bots.tools.config_tools:ListTemplatesTool"
        )
        from dataknobs_bots.tools.config_tools import ListTemplatesTool

        assert result is ListTemplatesTool

    def test_dot_separator_fallback(self) -> None:
        result = resolve_callable(
            "dataknobs_bots.tools.config_tools.ListTemplatesTool"
        )
        from dataknobs_bots.tools.config_tools import ListTemplatesTool

        assert result is ListTemplatesTool

    def test_import_error(self) -> None:
        with pytest.raises(ImportError):
            resolve_callable("nonexistent.module:Foo")

    def test_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            resolve_callable("dataknobs_bots.tools.config_tools:NoSuchClass")

    def test_not_callable_error(self) -> None:
        with pytest.raises(ValueError, match="not callable"):
            resolve_callable("dataknobs_bots.tools.config_tools:logger")


class TestListTemplatesToolFromConfig:
    """Tests for ListTemplatesTool.from_config()."""

    def test_from_config_with_dir(self, tmp_path: Path) -> None:
        from dataknobs_bots.tools.config_tools import ListTemplatesTool

        # Create a minimal template file
        template = {
            "name": "test_template",
            "description": "A test template",
            "version": "1.0.0",
            "tags": ["test"],
            "variables": [],
            "structure": {"bot": {"llm": {"provider": "echo"}}},
        }
        (tmp_path / "test_template.yaml").write_text(
            yaml.dump(template), encoding="utf-8"
        )

        tool = ListTemplatesTool.from_config(
            {"template_dir": str(tmp_path)}
        )
        assert tool.name == "list_templates"
        assert tool._registry is not None

    def test_from_config_missing_dir(self) -> None:
        from dataknobs_bots.tools.config_tools import ListTemplatesTool

        tool = ListTemplatesTool.from_config(
            {"template_dir": "/nonexistent/path"}
        )
        assert tool.name == "list_templates"


class TestGetTemplateDetailsToolFromConfig:
    """Tests for GetTemplateDetailsTool.from_config()."""

    def test_from_config(self, tmp_path: Path) -> None:
        from dataknobs_bots.tools.config_tools import GetTemplateDetailsTool

        tool = GetTemplateDetailsTool.from_config(
            {"template_dir": str(tmp_path)}
        )
        assert tool.name == "get_template_details"


class TestPreviewConfigToolFromConfig:
    """Tests for PreviewConfigTool.from_config()."""

    def test_from_config(self) -> None:
        from dataknobs_bots.tools.config_tools import PreviewConfigTool

        tool = PreviewConfigTool.from_config(
            {
                "builder_factory": "dataknobs_bots.config.builder:DynaBotConfigBuilder",
            }
        )
        assert tool.name == "preview_config"
        assert tool._builder_factory is not None

    def test_from_config_missing_factory_raises(self) -> None:
        from dataknobs_bots.tools.config_tools import PreviewConfigTool

        with pytest.raises(KeyError):
            PreviewConfigTool.from_config({})


class TestValidateConfigToolFromConfig:
    """Tests for ValidateConfigTool.from_config()."""

    def test_from_config_minimal(self) -> None:
        from dataknobs_bots.tools.config_tools import ValidateConfigTool

        tool = ValidateConfigTool.from_config({})
        assert tool.name == "validate_config"
        assert tool._validator is not None
        assert tool._builder_factory is None

    def test_from_config_with_factory(self) -> None:
        from dataknobs_bots.tools.config_tools import ValidateConfigTool

        tool = ValidateConfigTool.from_config(
            {
                "builder_factory": "dataknobs_bots.config.builder:DynaBotConfigBuilder",
            }
        )
        assert tool._builder_factory is not None


class TestSaveConfigToolFromConfig:
    """Tests for SaveConfigTool.from_config()."""

    def test_from_config_minimal(self, tmp_path: Path) -> None:
        from dataknobs_bots.tools.config_tools import SaveConfigTool

        tool = SaveConfigTool.from_config(
            {"config_dir": str(tmp_path)}
        )
        assert tool.name == "save_config"
        assert tool._draft_manager.output_dir == tmp_path
        assert tool._on_save is None
        assert tool._builder_factory is None
        assert tool._portable is False

    def test_from_config_with_portable(self, tmp_path: Path) -> None:
        from dataknobs_bots.tools.config_tools import SaveConfigTool

        tool = SaveConfigTool.from_config(
            {"config_dir": str(tmp_path), "portable": True}
        )
        assert tool._portable is True


class TestResolveToolFromConfigIntegration:
    """Test that _resolve_tool() uses from_config() when available."""

    def test_uses_from_config_when_present(self, tmp_path: Path) -> None:
        from dataknobs_bots.bot.base import DynaBot

        tool_config: dict[str, Any] = {
            "class": "dataknobs_bots.tools.config_tools.ListTemplatesTool",
            "params": {"template_dir": str(tmp_path)},
        }
        tool = DynaBot._resolve_tool(tool_config, {})
        assert tool is not None
        assert tool.name == "list_templates"

    def test_falls_back_to_init(self) -> None:
        from dataknobs_bots.bot.base import DynaBot

        tool_config: dict[str, Any] = {
            "class": "dataknobs_bots.tools.config_tools.ListAvailableToolsTool",
            "params": {"available_tools": [{"name": "test", "description": "Test"}]},
        }
        tool = DynaBot._resolve_tool(tool_config, {})
        assert tool is not None
        assert tool.name == "list_available_tools"
