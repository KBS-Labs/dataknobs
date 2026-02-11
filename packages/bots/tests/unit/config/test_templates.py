"""Tests for config/templates.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from dataknobs_bots.config.templates import (
    ConfigTemplate,
    ConfigTemplateRegistry,
    TemplateVariable,
)


class TestTemplateVariable:
    """Tests for TemplateVariable dataclass."""

    def test_to_dict_minimal(self) -> None:
        var = TemplateVariable(name="subject")
        d = var.to_dict()
        assert d["name"] == "subject"
        assert d["type"] == "string"
        assert d["required"] is False

    def test_to_dict_full(self) -> None:
        var = TemplateVariable(
            name="mode",
            description="Operating mode",
            type="enum",
            required=True,
            default="quiz",
            choices=["quiz", "tutor"],
            validation={"pattern": "^[a-z]+$"},
        )
        d = var.to_dict()
        assert d["name"] == "mode"
        assert d["description"] == "Operating mode"
        assert d["choices"] == ["quiz", "tutor"]
        assert d["validation"] == {"pattern": "^[a-z]+$"}

    def test_from_dict_round_trip(self) -> None:
        original = TemplateVariable(
            name="test",
            description="A test var",
            type="integer",
            required=True,
            default=42,
            choices=[1, 2, 42],
        )
        restored = TemplateVariable.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.type == original.type
        assert restored.required == original.required
        assert restored.default == original.default
        assert restored.choices == original.choices

    def test_from_dict_defaults(self) -> None:
        var = TemplateVariable.from_dict({"name": "x"})
        assert var.type == "string"
        assert var.required is False
        assert var.default is None
        assert var.choices is None


class TestConfigTemplate:
    """Tests for ConfigTemplate dataclass."""

    def test_get_required_variables(self) -> None:
        template = ConfigTemplate(
            name="test",
            variables=[
                TemplateVariable(name="a", required=True),
                TemplateVariable(name="b", required=False),
                TemplateVariable(name="c", required=True),
            ],
        )
        required = template.get_required_variables()
        assert len(required) == 2
        assert [v.name for v in required] == ["a", "c"]

    def test_get_optional_variables(self) -> None:
        template = ConfigTemplate(
            name="test",
            variables=[
                TemplateVariable(name="a", required=True),
                TemplateVariable(name="b", required=False),
            ],
        )
        optional = template.get_optional_variables()
        assert len(optional) == 1
        assert optional[0].name == "b"

    def test_to_dict_round_trip(self) -> None:
        original = ConfigTemplate(
            name="my_template",
            description="A template",
            version="2.0.0",
            tags=["test", "example"],
            variables=[TemplateVariable(name="var1", required=True)],
            structure={"bot": {"llm": {"provider": "{{provider}}"}}},
        )
        restored = ConfigTemplate.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.version == original.version
        assert restored.tags == original.tags
        assert len(restored.variables) == 1
        assert restored.variables[0].name == "var1"
        assert restored.structure == original.structure

    def test_from_yaml_file(self, tmp_path: Path) -> None:
        template_data = {
            "name": "test_template",
            "description": "From YAML",
            "version": "1.0.0",
            "tags": ["test"],
            "variables": [
                {"name": "bot_name", "type": "string", "required": True},
            ],
            "structure": {"bot": {"system_prompt": "Hello {{bot_name}}"}},
        }
        path = tmp_path / "test-template.yaml"
        with open(path, "w") as f:
            yaml.dump(template_data, f)

        template = ConfigTemplate.from_yaml_file(path)
        assert template.name == "test_template"
        assert template.description == "From YAML"
        assert len(template.variables) == 1

    def test_from_yaml_file_infers_name(self, tmp_path: Path) -> None:
        path = tmp_path / "my-bot.yaml"
        with open(path, "w") as f:
            yaml.dump({"description": "no name"}, f)

        template = ConfigTemplate.from_yaml_file(path)
        # Hyphens are converted to underscores for internal names
        assert template.name == "my_bot"


class TestConfigTemplateRegistry:
    """Tests for ConfigTemplateRegistry."""

    def _make_template(
        self, name: str, tags: list[str] | None = None
    ) -> ConfigTemplate:
        return ConfigTemplate(
            name=name,
            description=f"Template {name}",
            tags=tags or [],
            variables=[
                TemplateVariable(name="bot_name", required=True),
            ],
            structure={"bot": {"system_prompt": "I am {{bot_name}}"}},
        )

    def test_register_and_get(self) -> None:
        registry = ConfigTemplateRegistry()
        template = self._make_template("test")
        registry.register(template)
        assert registry.get("test") is template
        assert registry.get("nonexistent") is None

    def test_list_templates(self) -> None:
        registry = ConfigTemplateRegistry()
        registry.register(self._make_template("a", ["tag1"]))
        registry.register(self._make_template("b", ["tag1", "tag2"]))
        registry.register(self._make_template("c", ["tag2"]))

        all_templates = registry.list_templates()
        assert len(all_templates) == 3

        tag1 = registry.list_templates(tags=["tag1"])
        assert len(tag1) == 2

        both_tags = registry.list_templates(tags=["tag1", "tag2"])
        assert len(both_tags) == 1
        assert both_tags[0].name == "b"

    def test_load_from_directory(self, tmp_path: Path) -> None:
        for name in ["alpha", "beta", "gamma"]:
            data = {
                "name": name,
                "description": f"Template {name}",
                "variables": [],
                "structure": {"bot": {"llm": {"provider": "ollama"}}},
            }
            path = tmp_path / f"{name}.yaml"
            with open(path, "w") as f:
                yaml.dump(data, f)

        # Should skip README and base
        readme_path = tmp_path / "README.yaml"
        with open(readme_path, "w") as f:
            yaml.dump({"name": "readme"}, f)
        base_path = tmp_path / "base.yaml"
        with open(base_path, "w") as f:
            yaml.dump({"name": "base"}, f)

        registry = ConfigTemplateRegistry()
        count = registry.load_from_directory(tmp_path)
        assert count == 3
        assert registry.get("alpha") is not None
        assert registry.get("readme") is None
        assert registry.get("base") is None

    def test_apply_template_substitution(self) -> None:
        registry = ConfigTemplateRegistry()
        template = ConfigTemplate(
            name="test",
            variables=[
                TemplateVariable(name="bot_name", required=True),
                TemplateVariable(name="temp", default=0.7),
            ],
            structure={
                "bot": {
                    "system_prompt": "I am {{bot_name}}",
                    "llm": {"temperature": "{{temp}}"},
                }
            },
        )
        registry.register(template)

        result = registry.apply_template("test", {"bot_name": "Helper"})
        assert result["bot"]["system_prompt"] == "I am Helper"

    def test_apply_template_uses_defaults(self) -> None:
        registry = ConfigTemplateRegistry()
        template = ConfigTemplate(
            name="test",
            variables=[
                TemplateVariable(name="mode", default="chat"),
            ],
            structure={"mode": "{{mode}}"},
        )
        registry.register(template)

        result = registry.apply_template("test", {})
        assert result["mode"] == "chat"

    def test_apply_template_not_found(self) -> None:
        registry = ConfigTemplateRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.apply_template("nonexistent", {})

    def test_validate_variables_valid(self) -> None:
        registry = ConfigTemplateRegistry()
        template = ConfigTemplate(
            name="test",
            variables=[
                TemplateVariable(name="name", required=True),
                TemplateVariable(name="mode", choices=["a", "b"]),
            ],
        )
        registry.register(template)

        result = registry.validate_variables("test", {"name": "bot", "mode": "a"})
        assert result.valid is True

    def test_validate_variables_missing_required(self) -> None:
        registry = ConfigTemplateRegistry()
        template = ConfigTemplate(
            name="test",
            variables=[TemplateVariable(name="name", required=True)],
        )
        registry.register(template)

        result = registry.validate_variables("test", {})
        assert result.valid is False
        assert any("name" in e for e in result.errors)

    def test_validate_variables_invalid_choice(self) -> None:
        registry = ConfigTemplateRegistry()
        template = ConfigTemplate(
            name="test",
            variables=[
                TemplateVariable(name="mode", choices=["a", "b"]),
            ],
        )
        registry.register(template)

        result = registry.validate_variables("test", {"mode": "invalid"})
        assert result.valid is False
        assert any("invalid" in e for e in result.errors)

    def test_validate_variables_template_not_found(self) -> None:
        registry = ConfigTemplateRegistry()
        result = registry.validate_variables("missing", {})
        assert result.valid is False

    def test_load_from_file(self, tmp_path: Path) -> None:
        data: dict[str, Any] = {
            "name": "loaded",
            "description": "Loaded from file",
            "variables": [],
            "structure": {},
        }
        path = tmp_path / "loaded.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)

        registry = ConfigTemplateRegistry()
        template = registry.load_from_file(path)
        assert template.name == "loaded"
        assert registry.get("loaded") is template
