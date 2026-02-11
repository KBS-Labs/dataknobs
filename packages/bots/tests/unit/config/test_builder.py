"""Tests for config/builder.py."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from dataknobs_bots.config.builder import DynaBotConfigBuilder
from dataknobs_bots.config.templates import ConfigTemplate, TemplateVariable


class TestDynaBotConfigBuilder:
    """Tests for DynaBotConfigBuilder."""

    def test_minimal_build(self) -> None:
        config = (
            DynaBotConfigBuilder()
            .set_llm("ollama", model="llama3.2")
            .set_conversation_storage("memory")
            .build()
        )
        assert config["llm"]["provider"] == "ollama"
        assert config["llm"]["model"] == "llama3.2"
        assert config["conversation_storage"]["backend"] == "memory"

    def test_build_with_all_components(self) -> None:
        config = (
            DynaBotConfigBuilder()
            .set_llm("openai", model="gpt-4", temperature=0.5)
            .set_conversation_storage("sqlite", path="test.db")
            .set_memory("buffer", max_messages=100)
            .set_reasoning("react", max_iterations=5)
            .set_system_prompt(content="You are a helper.")
            .set_knowledge_base(enabled=True, type="rag")
            .add_tool("my_module.MyTool", param1="val1")
            .add_middleware("my_module.MyMiddleware")
            .build()
        )
        assert config["llm"]["provider"] == "openai"
        assert config["llm"]["temperature"] == 0.5
        assert config["conversation_storage"]["backend"] == "sqlite"
        assert config["memory"]["type"] == "buffer"
        assert config["memory"]["max_messages"] == 100
        assert config["reasoning"]["strategy"] == "react"
        assert config["system_prompt"] == "You are a helper."
        assert config["knowledge_base"]["enabled"] is True
        assert len(config["tools"]) == 1
        assert config["tools"][0]["class"] == "my_module.MyTool"
        assert config["tools"][0]["params"]["param1"] == "val1"
        assert len(config["middleware"]) == 1

    def test_set_llm_resource(self) -> None:
        config = (
            DynaBotConfigBuilder()
            .set_llm_resource("default", temperature=0.3)
            .set_conversation_storage("memory")
            .build()
        )
        assert config["llm"]["$resource"] == "default"
        assert config["llm"]["type"] == "llm_providers"
        assert config["llm"]["temperature"] == 0.3

    def test_set_conversation_storage_resource(self) -> None:
        config = (
            DynaBotConfigBuilder()
            .set_llm("ollama")
            .set_conversation_storage_resource("conversations")
            .build()
        )
        assert config["conversation_storage"]["$resource"] == "conversations"
        assert config["conversation_storage"]["type"] == "databases"

    def test_system_prompt_string(self) -> None:
        config = (
            DynaBotConfigBuilder()
            .set_llm("ollama")
            .set_conversation_storage("memory")
            .set_system_prompt(content="Hello world")
            .build()
        )
        assert config["system_prompt"] == "Hello world"

    def test_system_prompt_dict(self) -> None:
        config = (
            DynaBotConfigBuilder()
            .set_llm("ollama")
            .set_conversation_storage("memory")
            .set_system_prompt(name="my_template")
            .build()
        )
        assert config["system_prompt"]["name"] == "my_template"

    def test_system_prompt_with_rag(self) -> None:
        config = (
            DynaBotConfigBuilder()
            .set_llm("ollama")
            .set_conversation_storage("memory")
            .set_system_prompt(
                content="Help me",
                rag_configs=[{"name": "kb", "max_results": 5}],
            )
            .build()
        )
        assert config["system_prompt"]["content"] == "Help me"
        assert len(config["system_prompt"]["rag_configs"]) == 1

    def test_custom_section(self) -> None:
        config = (
            DynaBotConfigBuilder()
            .set_llm("ollama")
            .set_conversation_storage("memory")
            .set_custom_section("educational", {"mode": "tutor"})
            .build()
        )
        assert config["educational"]["mode"] == "tutor"

    def test_multiple_tools(self) -> None:
        config = (
            DynaBotConfigBuilder()
            .set_llm("ollama")
            .set_conversation_storage("memory")
            .add_tool("tool.A")
            .add_tool("tool.B", x=1)
            .build()
        )
        assert len(config["tools"]) == 2
        assert config["tools"][0]["class"] == "tool.A"
        assert "params" not in config["tools"][0]
        assert config["tools"][1]["params"]["x"] == 1

    def test_build_portable(self) -> None:
        portable = (
            DynaBotConfigBuilder()
            .set_llm_resource("default")
            .set_conversation_storage_resource()
            .set_custom_section("domain", {"id": "test-bot"})
            .build_portable()
        )
        assert "bot" in portable
        assert portable["bot"]["llm"]["$resource"] == "default"
        assert "domain" in portable
        assert portable["domain"]["id"] == "test-bot"
        # Custom sections should NOT be inside bot
        assert "domain" not in portable["bot"]

    def test_build_fails_without_required(self) -> None:
        builder = DynaBotConfigBuilder()
        with pytest.raises(ValueError, match="validation failed"):
            builder.build()

    def test_validate_returns_result(self) -> None:
        builder = (
            DynaBotConfigBuilder()
            .set_llm("ollama")
            .set_conversation_storage("memory")
        )
        result = builder.validate()
        assert result.valid is True

    def test_validate_incomplete(self) -> None:
        builder = DynaBotConfigBuilder()
        result = builder.validate()
        assert result.valid is False

    def test_to_yaml(self) -> None:
        yaml_str = (
            DynaBotConfigBuilder()
            .set_llm("ollama", model="llama3.2")
            .set_conversation_storage("memory")
            .to_yaml()
        )
        parsed = yaml.safe_load(yaml_str)
        assert "bot" in parsed
        assert parsed["bot"]["llm"]["provider"] == "ollama"

    def test_reset(self) -> None:
        builder = (
            DynaBotConfigBuilder()
            .set_llm("ollama")
            .set_conversation_storage("memory")
        )
        builder.reset()
        result = builder.validate()
        assert result.valid is False

    def test_from_config_flat(self) -> None:
        original: dict[str, Any] = {
            "llm": {"provider": "ollama"},
            "conversation_storage": {"backend": "memory"},
            "system_prompt": "Hello",
        }
        builder = DynaBotConfigBuilder.from_config(original)
        config = builder.build()
        assert config["llm"]["provider"] == "ollama"
        assert config["system_prompt"] == "Hello"

    def test_from_config_portable(self) -> None:
        original: dict[str, Any] = {
            "bot": {
                "llm": {"$resource": "default", "type": "llm_providers"},
                "conversation_storage": {"$resource": "db", "type": "databases"},
            },
            "domain": {"id": "my-bot"},
        }
        builder = DynaBotConfigBuilder.from_config(original)
        portable = builder.build_portable()
        assert portable["bot"]["llm"]["$resource"] == "default"
        assert portable["domain"]["id"] == "my-bot"

    def test_from_template(self) -> None:
        template = ConfigTemplate(
            name="test",
            variables=[
                TemplateVariable(name="bot_name", required=True),
                TemplateVariable(name="temp", default=0.7),
            ],
            structure={
                "bot": {
                    "llm": {
                        "$resource": "default",
                        "type": "llm_providers",
                        "temperature": "{{temp}}",
                    },
                    "conversation_storage": {
                        "$resource": "conversations",
                        "type": "databases",
                    },
                    "system_prompt": "I am {{bot_name}}",
                },
            },
        )
        builder = DynaBotConfigBuilder().from_template(
            template, {"bot_name": "Helper"}
        )
        config = builder.build()
        assert config["system_prompt"] == "I am Helper"
        assert config["llm"]["$resource"] == "default"

    def test_from_template_with_custom_sections(self) -> None:
        template = ConfigTemplate(
            name="test",
            variables=[TemplateVariable(name="mode", default="quiz")],
            structure={
                "bot": {
                    "llm": {"provider": "ollama"},
                    "conversation_storage": {"backend": "memory"},
                },
                "educational": {"mode": "{{mode}}"},
            },
        )
        builder = DynaBotConfigBuilder().from_template(template, {"mode": "tutor"})
        portable = builder.build_portable()
        assert portable["educational"]["mode"] == "tutor"
        assert "educational" not in portable["bot"]

    def test_merge_overrides(self) -> None:
        config = (
            DynaBotConfigBuilder()
            .set_llm("ollama", model="llama3.2", temperature=0.7)
            .set_conversation_storage("memory")
            .merge_overrides({"llm": {"temperature": 0.3}})
            .build()
        )
        assert config["llm"]["temperature"] == 0.3
        assert config["llm"]["provider"] == "ollama"
        assert config["llm"]["model"] == "llama3.2"

    def test_method_chaining(self) -> None:
        builder = DynaBotConfigBuilder()
        result = builder.set_llm("ollama")
        assert result is builder
        result = builder.set_conversation_storage("memory")
        assert result is builder
        result = builder.set_memory("buffer")
        assert result is builder
        result = builder.set_reasoning("simple")
        assert result is builder
        result = builder.add_tool("tool.X")
        assert result is builder
        result = builder.add_middleware("mw.Y")
        assert result is builder
        result = builder.set_custom_section("x", {})
        assert result is builder
        result = builder.reset()
        assert result is builder
