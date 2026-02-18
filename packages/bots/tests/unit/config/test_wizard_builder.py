"""Tests for config/wizard_builder.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from dataknobs_bots.config.wizard_builder import (
    StageConfig,
    TransitionConfig,
    WizardConfig,
    WizardConfigBuilder,
)


class TestWizardConfigBuilder:
    """Tests for WizardConfigBuilder."""

    def test_minimal_build(self) -> None:
        wizard = (
            WizardConfigBuilder("test-wizard")
            .add_structured_stage("start", "Hello", is_start=True, is_end=True)
            .build()
        )
        assert wizard.name == "test-wizard"
        assert wizard.version == "1.0"
        assert len(wizard.stages) == 1
        assert wizard.stages[0].name == "start"
        assert wizard.stages[0].is_start is True

    def test_conversation_stage_sets_mode(self) -> None:
        wizard = (
            WizardConfigBuilder("test")
            .add_conversation_stage("chat", "Talk to me", is_start=True, is_end=True)
            .build()
        )
        assert wizard.stages[0].mode == "conversation"

    def test_end_stage_sets_is_end(self) -> None:
        wizard = (
            WizardConfigBuilder("test")
            .add_structured_stage("start", "Begin", is_start=True)
            .add_end_stage("done", "Goodbye")
            .add_transition("start", "done")
            .build()
        )
        end = next(s for s in wizard.stages if s.name == "done")
        assert end.is_end is True

    def test_structured_stage_with_all_options(self) -> None:
        wizard = (
            WizardConfigBuilder("test")
            .add_structured_stage(
                name="collect",
                prompt="Enter your name",
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
                tools=["validator"],
                is_start=True,
                is_end=True,
                can_skip=True,
                skip_default="Anonymous",
                suggestions=["Alice", "Bob"],
                response_template="Hello {{name}}",
                help_text="Type your name",
                reasoning="react",
                max_iterations=5,
                context_generation={"variables": {"greeting": "Generate a greeting"}},
            )
            .build()
        )
        stage = wizard.stages[0]
        assert stage.schema is not None
        assert stage.tools == ("validator",)
        assert stage.can_skip is True
        assert stage.skip_default == "Anonymous"
        assert stage.suggestions == ("Alice", "Bob")
        assert stage.response_template == "Hello {{name}}"
        assert stage.help_text == "Type your name"
        assert stage.reasoning == "react"
        assert stage.max_iterations == 5
        assert stage.context_generation is not None
        assert stage.context_generation.variables == {"greeting": "Generate a greeting"}

    def test_multiple_stages_with_transitions(self) -> None:
        wizard = (
            WizardConfigBuilder("flow")
            .add_structured_stage("step1", "First step", is_start=True)
            .add_structured_stage("step2", "Second step")
            .add_end_stage("done", "Complete")
            .add_transition("step1", "step2", condition="data.get('ready')")
            .add_transition("step2", "done")
            .build()
        )
        assert len(wizard.stages) == 3
        step1 = next(s for s in wizard.stages if s.name == "step1")
        assert len(step1.transitions) == 1
        assert step1.transitions[0].target == "step2"
        assert step1.transitions[0].condition == "data.get('ready')"

    def test_transition_with_all_options(self) -> None:
        wizard = (
            WizardConfigBuilder("test")
            .add_structured_stage("a", "Stage A", is_start=True)
            .add_end_stage("b", "Stage B")
            .add_transition(
                "a",
                "b",
                condition="data.get('x')",
                transform="my_transform",
                priority=10,
                derive={"field": "expression"},
                metadata={"label": "go to B"},
            )
            .build()
        )
        t = wizard.stages[0].transitions[0]
        assert t.condition == "data.get('x')"
        assert t.transform == "my_transform"
        assert t.priority == 10
        assert t.derive == {"field": "expression"}
        assert t.metadata == {"label": "go to B"}

    def test_conversation_start_factory(self) -> None:
        builder = WizardConfigBuilder.conversation_start(
            name="tutor",
            prompt="Let's learn about {subject}.",
            tools=["knowledge_search", "calculator"],
            tool_reasoning="react",
            max_tool_iterations=5,
        )
        wizard = builder.build()
        assert wizard.name == "tutor"
        assert wizard.version == "1.0.0"
        assert wizard.settings["tool_reasoning"] == "react"
        assert wizard.settings["max_tool_iterations"] == 5
        assert len(wizard.stages) == 1
        stage = wizard.stages[0]
        assert stage.name == "conversation"
        assert stage.mode == "conversation"
        assert stage.is_start is True
        assert stage.tools == ("knowledge_search", "calculator")

    def test_intent_detection(self) -> None:
        wizard = (
            WizardConfigBuilder("test")
            .add_conversation_stage("chat", "Talk", is_start=True, is_end=True)
            .add_intent_detection(
                "chat",
                method="keyword",
                intents=[{"id": "quiz", "keywords": ["quiz", "test me"]}],
            )
            .build()
        )
        stage = wizard.stages[0]
        assert stage.intent_detection is not None
        assert stage.intent_detection.method == "keyword"
        assert len(stage.intent_detection.intents) == 1
        assert stage.intent_detection.intents[0]["id"] == "quiz"

    def test_intent_detection_via_conversation_stage(self) -> None:
        wizard = (
            WizardConfigBuilder("test")
            .add_conversation_stage(
                "chat",
                "Talk",
                is_start=True,
                is_end=True,
                intent_detection={
                    "method": "llm",
                    "intents": [{"id": "help", "description": "Needs help"}],
                },
            )
            .build()
        )
        assert wizard.stages[0].intent_detection is not None
        assert wizard.stages[0].intent_detection.method == "llm"

    def test_global_tasks(self) -> None:
        wizard = (
            WizardConfigBuilder("test")
            .add_structured_stage("start", "Go", is_start=True, is_end=True)
            .add_global_task(
                "validate",
                "Validate all inputs",
                required=True,
            )
            .add_global_task(
                "save",
                "Save results",
                depends_on=["validate"],
                completed_by="tool_result",
                tool_name="save_tool",
            )
            .build()
        )
        assert len(wizard.global_tasks) == 2
        assert wizard.global_tasks[0]["id"] == "validate"
        assert wizard.global_tasks[1]["depends_on"] == ["validate"]

    def test_settings(self) -> None:
        wizard = (
            WizardConfigBuilder("test")
            .set_settings(tool_reasoning="react", timeout_seconds=300)
            .add_structured_stage("start", "Go", is_start=True, is_end=True)
            .build()
        )
        assert wizard.settings["tool_reasoning"] == "react"
        assert wizard.settings["timeout_seconds"] == 300

    def test_metadata(self) -> None:
        wizard = (
            WizardConfigBuilder("my-wizard")
            .set_version("2.0.0")
            .set_description("A test wizard")
            .add_structured_stage("start", "Go", is_start=True, is_end=True)
            .build()
        )
        assert wizard.version == "2.0.0"
        assert wizard.description == "A test wizard"

    def test_method_chaining(self) -> None:
        builder = WizardConfigBuilder("test")
        assert builder.set_version("1.0") is builder
        assert builder.set_description("desc") is builder
        assert builder.set_settings(x=1) is builder
        assert builder.add_structured_stage("s", "p", is_start=True, is_end=True) is builder
        assert builder.add_conversation_stage("c", "p") is builder
        assert builder.add_end_stage("e", "p") is builder
        assert builder.add_transition("c", "e") is builder
        assert builder.add_intent_detection("c") is builder
        assert builder.add_global_task("t", "desc") is builder

    def test_add_raw_stage(self) -> None:
        stage = StageConfig(name="raw", prompt="Raw stage", is_start=True, is_end=True)
        wizard = WizardConfigBuilder("test").add_stage(stage).build()
        assert wizard.stages[0].name == "raw"


class TestWizardConfigBuilderValidation:
    """Tests for validation logic."""

    def test_no_stages_error(self) -> None:
        result = WizardConfigBuilder("test").validate()
        assert result.valid is False
        assert any("at least one stage" in e for e in result.errors)

    def test_no_start_stage_error(self) -> None:
        builder = WizardConfigBuilder("test")
        builder._stages.append(
            StageConfig(name="orphan", prompt="No start", is_end=True)
        )
        result = builder.validate()
        assert result.valid is False
        assert any("start stage" in e for e in result.errors)

    def test_multiple_start_stages_error(self) -> None:
        result = (
            WizardConfigBuilder("test")
            .add_structured_stage("a", "A", is_start=True, is_end=True)
            .add_structured_stage("b", "B", is_start=True, is_end=True)
            .validate()
        )
        assert result.valid is False
        assert any("multiple start" in e for e in result.errors)

    def test_duplicate_stage_names_error(self) -> None:
        builder = WizardConfigBuilder("test")
        builder._stages.append(
            StageConfig(name="dup", prompt="First", is_start=True, is_end=True)
        )
        builder._stages.append(
            StageConfig(name="dup", prompt="Second", is_end=True)
        )
        result = builder.validate()
        assert result.valid is False
        assert any("Duplicate stage name" in e for e in result.errors)

    def test_invalid_transition_target_error(self) -> None:
        result = (
            WizardConfigBuilder("test")
            .add_structured_stage("start", "Go", is_start=True)
            .add_transition("start", "nonexistent")
            .validate()
        )
        assert result.valid is False
        assert any("unknown stage 'nonexistent'" in e for e in result.errors)

    def test_transition_from_unknown_source_error(self) -> None:
        result = (
            WizardConfigBuilder("test")
            .add_structured_stage("start", "Go", is_start=True, is_end=True)
            .add_transition("ghost", "start")
            .validate()
        )
        assert result.valid is False
        assert any("unknown stage 'ghost'" in e for e in result.errors)

    def test_intent_detection_unknown_stage_error(self) -> None:
        result = (
            WizardConfigBuilder("test")
            .add_structured_stage("start", "Go", is_start=True, is_end=True)
            .add_intent_detection("nonexistent")
            .validate()
        )
        assert result.valid is False
        assert any("unknown stage 'nonexistent'" in e for e in result.errors)

    def test_invalid_reasoning_error(self) -> None:
        result = (
            WizardConfigBuilder("test")
            .add_structured_stage(
                "start", "Go", is_start=True, is_end=True, reasoning="invalid"
            )
            .validate()
        )
        assert result.valid is False
        assert any("invalid reasoning" in e for e in result.errors)

    def test_invalid_mode_error(self) -> None:
        builder = WizardConfigBuilder("test")
        builder._stages.append(
            StageConfig(name="bad", prompt="Bad mode", is_start=True, is_end=True, mode="invalid")
        )
        result = builder.validate()
        assert result.valid is False
        assert any("invalid mode" in e for e in result.errors)

    def test_orphan_stage_warning(self) -> None:
        result = (
            WizardConfigBuilder("test")
            .add_structured_stage("start", "Go", is_start=True, is_end=True)
            .add_structured_stage("orphan", "Unreachable", is_end=True)
            .validate()
        )
        assert result.valid is True
        assert any("not reachable" in w for w in result.warnings)

    def test_end_stage_with_transitions_warning(self) -> None:
        result = (
            WizardConfigBuilder("test")
            .add_structured_stage("start", "Go", is_start=True)
            .add_end_stage("done", "Done")
            .add_transition("start", "done")
            .add_transition("done", "start")
            .validate()
        )
        assert result.valid is True
        assert any("never be followed" in w for w in result.warnings)

    def test_max_iterations_without_reasoning_warning(self) -> None:
        result = (
            WizardConfigBuilder("test")
            .add_structured_stage(
                "start", "Go", is_start=True, is_end=True, max_iterations=5
            )
            .validate()
        )
        assert result.valid is True
        assert any("max_iterations" in w for w in result.warnings)

    def test_build_raises_on_invalid(self) -> None:
        with pytest.raises(ValueError, match="validation failed"):
            WizardConfigBuilder("test").build()

    def test_subflow_target_allowed(self) -> None:
        builder = WizardConfigBuilder("test")
        builder._stages.append(
            StageConfig(
                name="start",
                prompt="Go",
                is_start=True,
                is_end=True,
                transitions=(
                    TransitionConfig(target="_subflow", subflow={"network": "sub"}),
                ),
            )
        )
        result = builder.validate()
        assert result.valid is True


class TestWizardConfigSerialization:
    """Tests for serialization and roundtrip."""

    def _build_multi_stage(self) -> WizardConfig:
        return (
            WizardConfigBuilder("roundtrip-test")
            .set_version("2.0.0")
            .set_description("A roundtrip test wizard")
            .set_settings(tool_reasoning="react")
            .add_conversation_stage(
                "chat",
                "Talk to me",
                tools=["search"],
                is_start=True,
                suggestions=["Hello"],
            )
            .add_structured_stage(
                "collect",
                "Enter data",
                schema={"type": "object", "properties": {"x": {"type": "integer"}}},
                can_skip=True,
                skip_default=0,
            )
            .add_end_stage("done", "Goodbye")
            .add_transition("chat", "collect", condition="data.get('ready')")
            .add_transition("collect", "done")
            .add_global_task("validate", "Validate inputs")
            .build()
        )

    def test_to_dict_structure(self) -> None:
        wizard = self._build_multi_stage()
        d = wizard.to_dict()
        assert d["name"] == "roundtrip-test"
        assert d["version"] == "2.0.0"
        assert d["description"] == "A roundtrip test wizard"
        assert d["settings"]["tool_reasoning"] == "react"
        assert len(d["stages"]) == 3
        assert d["stages"][0]["mode"] == "conversation"
        assert d["stages"][0]["tools"] == ["search"]
        assert d["stages"][1]["can_skip"] is True
        assert d["stages"][2]["is_end"] is True
        assert len(d["global_tasks"]) == 1

    def test_to_yaml(self) -> None:
        wizard = self._build_multi_stage()
        yaml_str = wizard.to_yaml()
        parsed = yaml.safe_load(yaml_str)
        assert parsed["name"] == "roundtrip-test"
        assert len(parsed["stages"]) == 3

    def test_to_file(self, tmp_path: Path) -> None:
        wizard = self._build_multi_stage()
        file_path = tmp_path / "subdir" / "wizard.yaml"
        wizard.to_file(file_path)
        assert file_path.exists()
        parsed = yaml.safe_load(file_path.read_text())
        assert parsed["name"] == "roundtrip-test"

    def test_dict_roundtrip(self) -> None:
        wizard1 = self._build_multi_stage()
        d1 = wizard1.to_dict()
        builder2 = WizardConfigBuilder.from_dict(d1)
        wizard2 = builder2.build()
        d2 = wizard2.to_dict()
        assert d1 == d2

    def test_file_roundtrip(self, tmp_path: Path) -> None:
        wizard1 = self._build_multi_stage()
        file_path = tmp_path / "wizard.yaml"
        wizard1.to_file(file_path)
        builder2 = WizardConfigBuilder.from_file(file_path)
        wizard2 = builder2.build()
        assert wizard1.to_dict() == wizard2.to_dict()

    def test_to_dict_omits_defaults(self) -> None:
        """Default values should not clutter the output dict."""
        wizard = (
            WizardConfigBuilder("minimal")
            .add_structured_stage("start", "Go", is_start=True, is_end=True)
            .build()
        )
        stage_dict = wizard.to_dict()["stages"][0]
        # Default-valued fields should be omitted
        assert "can_skip" not in stage_dict
        assert "suggestions" not in stage_dict
        assert "tools" not in stage_dict
        assert "help_text" not in stage_dict
        assert "schema" not in stage_dict
        assert "reasoning" not in stage_dict
        assert "mode" not in stage_dict
        # Explicitly set fields should be present
        assert stage_dict["is_start"] is True
        assert stage_dict["is_end"] is True

    def test_loader_roundtrip(self) -> None:
        """Built config can be consumed by WizardConfigLoader."""
        wizard = self._build_multi_stage()
        config_dict = wizard.to_dict()

        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config_dict)

        assert wizard_fsm.current_stage == "chat"
        assert wizard_fsm.stage_count == 3
        assert "search" in wizard_fsm.get_stage_tools("chat")


class TestDynaBotConfigBuilderIntegration:
    """Tests for set_reasoning_wizard() on DynaBotConfigBuilder."""

    def test_set_reasoning_wizard_with_path(self) -> None:
        from dataknobs_bots.config.builder import DynaBotConfigBuilder

        config = (
            DynaBotConfigBuilder()
            .set_llm("ollama")
            .set_conversation_storage("memory")
            .set_reasoning_wizard("configs/wizards/my-wizard.yaml")
            .build()
        )
        assert config["reasoning"]["strategy"] == "wizard"
        assert config["reasoning"]["wizard_config"] == "configs/wizards/my-wizard.yaml"

    def test_set_reasoning_wizard_with_config_object(self) -> None:
        from dataknobs_bots.config.builder import DynaBotConfigBuilder

        wizard = (
            WizardConfigBuilder("my-wizard")
            .add_structured_stage("start", "Go", is_start=True, is_end=True)
            .build()
        )
        config = (
            DynaBotConfigBuilder()
            .set_llm("ollama")
            .set_conversation_storage("memory")
            .set_reasoning_wizard(wizard, extraction_config={"scope": "stage"})
            .build()
        )
        assert config["reasoning"]["strategy"] == "wizard"
        assert config["reasoning"]["wizard_config"] == "my-wizard"
        assert config["reasoning"]["extraction_config"] == {"scope": "stage"}

    def test_set_reasoning_wizard_returns_self(self) -> None:
        from dataknobs_bots.config.builder import DynaBotConfigBuilder

        builder = DynaBotConfigBuilder()
        result = builder.set_reasoning_wizard("path.yaml")
        assert result is builder
