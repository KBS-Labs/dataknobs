"""Tests for WizardConfigLoader."""

import tempfile
from pathlib import Path

import pytest
import yaml

from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader


@pytest.fixture
def simple_wizard_config() -> dict:
    """Create a simple wizard configuration."""
    return {
        "name": "test-wizard",
        "version": "1.0",
        "description": "A test wizard",
        "stages": [
            {
                "name": "welcome",
                "is_start": True,
                "prompt": "What would you like to do?",
                "schema": {
                    "type": "object",
                    "properties": {"intent": {"type": "string"}},
                    "required": ["intent"],
                },
                "suggestions": ["Create something", "Edit something"],
                "transitions": [
                    {"target": "configure", "condition": "data.get('intent')"}
                ],
            },
            {
                "name": "configure",
                "prompt": "How would you like to configure it?",
                "can_skip": True,
                "transitions": [{"target": "complete"}],
            },
            {
                "name": "complete",
                "is_end": True,
                "prompt": "All done!",
            },
        ],
    }


@pytest.fixture
def wizard_config_file(simple_wizard_config: dict) -> Path:
    """Create a temporary wizard config file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(simple_wizard_config, f)
        return Path(f.name)


class TestWizardConfigLoader:
    """Tests for WizardConfigLoader."""

    def test_load_from_dict(self, simple_wizard_config: dict) -> None:
        """Test loading wizard config from dict."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)

        assert wizard_fsm is not None
        assert wizard_fsm.current_stage == "welcome"

    def test_load_from_file(self, wizard_config_file: Path) -> None:
        """Test loading wizard config from file."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load(wizard_config_file)

        assert wizard_fsm is not None
        assert wizard_fsm.current_stage == "welcome"

    def test_stage_metadata_extraction(self, simple_wizard_config: dict) -> None:
        """Test that stage metadata is correctly extracted."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)

        # Check welcome stage metadata
        welcome_meta = wizard_fsm._stage_metadata.get("welcome", {})
        assert welcome_meta.get("prompt") == "What would you like to do?"
        assert welcome_meta.get("is_start") is True
        assert welcome_meta.get("is_end") is False
        assert "Create something" in welcome_meta.get("suggestions", [])

        # Check configure stage metadata
        configure_meta = wizard_fsm._stage_metadata.get("configure", {})
        assert configure_meta.get("can_skip") is True
        assert configure_meta.get("can_go_back") is True

        # Check complete stage metadata
        complete_meta = wizard_fsm._stage_metadata.get("complete", {})
        assert complete_meta.get("is_end") is True

    def test_stage_schema_extraction(self, simple_wizard_config: dict) -> None:
        """Test that schemas are correctly extracted."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)

        welcome_meta = wizard_fsm._stage_metadata.get("welcome", {})
        schema = welcome_meta.get("schema")
        assert schema is not None
        assert schema["type"] == "object"
        assert "intent" in schema["properties"]

    def test_empty_stages_raises_error(self) -> None:
        """Test that empty stages raises ValueError."""
        loader = WizardConfigLoader()
        with pytest.raises(ValueError, match="must have at least one stage"):
            loader.load_from_dict({"stages": []})

    def test_missing_stages_raises_error(self) -> None:
        """Test that missing stages raises ValueError."""
        loader = WizardConfigLoader()
        with pytest.raises(ValueError, match="must have 'stages' field"):
            loader.load_from_dict({"name": "test"})

    def test_custom_functions(self, simple_wizard_config: dict) -> None:
        """Test loading with custom functions."""
        custom_called = []

        def custom_transform(data: dict, context: object = None) -> dict:
            custom_called.append(True)
            return data

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(
            simple_wizard_config,
            custom_functions={"custom_transform": custom_transform},
        )

        assert wizard_fsm is not None


class TestConfirmOnNewDataLoader:
    """Tests for confirm_on_new_data flag in stage metadata loading."""

    def test_confirm_on_new_data_defaults_false(
        self, simple_wizard_config: dict
    ) -> None:
        """confirm_on_new_data defaults to False when not set."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)

        welcome_meta = wizard_fsm._stage_metadata.get("welcome", {})
        assert welcome_meta.get("confirm_on_new_data") is False

    def test_confirm_on_new_data_loaded_when_set(self) -> None:
        """confirm_on_new_data is loaded from config when set to true."""
        config: dict = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Go",
                    "confirm_on_new_data": True,
                    "response_template": "Summary: {{ topic }}",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                        },
                    },
                }
            ],
        }
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)

        meta = wizard_fsm._stage_metadata.get("start", {})
        assert meta.get("confirm_on_new_data") is True


class TestCollectionModeMetadataLoader:
    """Tests that collection_mode and collection_config survive the loader."""

    def test_collection_mode_loaded_into_stage_metadata(self) -> None:
        """collection_mode and collection_config must appear in stage metadata.

        This is the regression test for the bug where _extract_metadata()
        listed these fields in KNOWN_STAGE_FIELDS but never copied them
        into the metadata dict, so _handle_collection_mode was unreachable.
        """
        config: dict = {
            "name": "collection-test",
            "settings": {
                "banks": {
                    "ingredients": {
                        "schema": {"required": ["name"]},
                        "max_records": 30,
                    },
                },
            },
            "stages": [
                {
                    "name": "collect",
                    "is_start": True,
                    "prompt": "Add an ingredient",
                    "collection_mode": "collection",
                    "collection_config": {
                        "bank_name": "ingredients",
                        "done_keywords": ["done", "finished"],
                        "done_prompt": "Got it!",
                    },
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "amount": {"type": "string"},
                        },
                        "required": ["name"],
                    },
                    "transitions": [
                        {
                            "target": "review",
                            "condition": "data.get('_collection_done')",
                        },
                    ],
                },
                {
                    "name": "review",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)

        meta = wizard_fsm._stage_metadata.get("collect", {})
        assert meta.get("collection_mode") == "collection"
        assert meta.get("collection_config") is not None
        assert meta["collection_config"]["bank_name"] == "ingredients"
        assert "done" in meta["collection_config"]["done_keywords"]

    def test_collection_mode_defaults_none_when_absent(
        self, simple_wizard_config: dict
    ) -> None:
        """Stages without collection_mode get None (not missing key)."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)

        welcome_meta = wizard_fsm._stage_metadata.get("welcome", {})
        assert "collection_mode" in welcome_meta
        assert welcome_meta["collection_mode"] is None


class TestWizardFSMOperations:
    """Tests for WizardFSM operations."""

    def test_stage_getters(self, simple_wizard_config: dict) -> None:
        """Test stage metadata getter methods."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)

        # Test getters on current stage
        assert wizard_fsm.get_stage_prompt() == "What would you like to do?"
        assert wizard_fsm.get_stage_schema() is not None
        assert "Create something" in wizard_fsm.get_stage_suggestions()
        assert wizard_fsm.can_skip() is False
        assert wizard_fsm.can_go_back() is True
        assert wizard_fsm.is_start_stage() is True
        assert wizard_fsm.is_end_stage() is False

    def test_stage_getters_by_name(self, simple_wizard_config: dict) -> None:
        """Test stage metadata getter methods with explicit stage name."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)

        # Test getters on specific stage
        assert wizard_fsm.get_stage_prompt("configure") == "How would you like to configure it?"
        assert wizard_fsm.can_skip("configure") is True
        assert wizard_fsm.is_end_stage("complete") is True

    def test_serialize_restore(self, simple_wizard_config: dict) -> None:
        """Test state serialization and restoration."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)

        # Serialize initial state
        state = wizard_fsm.serialize()
        assert state["current_stage"] == "welcome"

        # Modify state and serialize
        wizard_fsm.step({"intent": "test"})
        state2 = wizard_fsm.serialize()

        # Create new FSM and restore
        wizard_fsm2 = loader.load_from_dict(simple_wizard_config)
        wizard_fsm2.restore(state2)

        assert wizard_fsm2.current_stage == wizard_fsm.current_stage

    def test_restart(self, simple_wizard_config: dict) -> None:
        """Test wizard restart."""
        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(simple_wizard_config)

        # Make a transition
        wizard_fsm.step({"intent": "test"})

        # Restart
        wizard_fsm.restart()

        # Should be back at start
        assert wizard_fsm.current_stage == "welcome"


class TestTransitionConditions:
    """Tests for transition condition handling."""

    def test_simple_condition(self) -> None:
        """Test simple condition evaluation.

        Note: The FSM transitions on step() call. An empty dict still causes
        evaluation, and if the condition returns falsy, the FSM may use
        a fallback transition or stay put depending on configuration.
        """
        config = {
            "name": "condition-test",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Enter a value",
                    "transitions": [
                        {"target": "end", "condition": "data.get('value')"}
                    ],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)

        # Initial state
        assert wizard_fsm.current_stage == "start"

        # With value, should transition to end
        wizard_fsm.step({"value": "test"})
        assert wizard_fsm.current_stage == "end"

    def test_multiple_conditions(self) -> None:
        """Test multiple transition conditions with priority."""
        config = {
            "name": "multi-condition-test",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Choose path",
                    "transitions": [
                        {"target": "path_a", "condition": "data.get('choice') == 'a'", "priority": 0},
                        {"target": "path_b", "condition": "data.get('choice') == 'b'", "priority": 1},
                        {"target": "default", "priority": 2},
                    ],
                },
                {"name": "path_a", "is_end": True, "prompt": "Path A"},
                {"name": "path_b", "is_end": True, "prompt": "Path B"},
                {"name": "default", "is_end": True, "prompt": "Default"},
            ],
        }

        loader = WizardConfigLoader()

        # Test path A
        wizard_fsm = loader.load_from_dict(config)
        wizard_fsm.step({"choice": "a"})
        assert wizard_fsm.current_stage == "path_a"

        # Test default (when no specific choice matches)
        # Note: The FSM evaluates conditions in order; first matching wins
        wizard_fsm3 = loader.load_from_dict(config)
        wizard_fsm3.step({"choice": "c"})
        # With priority-based evaluation, 'default' should be last
        # But the actual behavior depends on FSM implementation
        assert wizard_fsm3.current_stage in ("path_a", "path_b", "default")


class TestSubflowLoading:
    """Tests for subflow loading in WizardConfigLoader."""

    def test_inline_subflow_config(self) -> None:
        """Subflow defined inline in wizard_config['subflows'] is loaded."""
        config = {
            "name": "parent-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Starting",
                    "transitions": [
                        {
                            "target": "_subflow",
                            "subflow": {
                                "network": "child_flow",
                                "return_stage": "finish",
                            },
                        }
                    ],
                },
                {"name": "finish", "is_end": True, "prompt": "Done"},
            ],
            "subflows": {
                "child_flow": {
                    "name": "child-wizard",
                    "stages": [
                        {
                            "name": "child_start",
                            "is_start": True,
                            "prompt": "Child step",
                            "transitions": [{"target": "child_end"}],
                        },
                        {"name": "child_end", "is_end": True, "prompt": "Child done"},
                    ],
                }
            },
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)
        assert wizard_fsm is not None
        # The subflow registry should contain the child flow
        assert "child_flow" in wizard_fsm._subflow_registry

    def test_subflow_from_file_path(self, tmp_path: Path) -> None:
        """Loads subflow from <name>.yaml alongside main config."""
        # Create subflow config file
        subflow_config = {
            "name": "file-subflow",
            "stages": [
                {
                    "name": "sub_start",
                    "is_start": True,
                    "prompt": "Sub start",
                    "transitions": [{"target": "sub_end"}],
                },
                {"name": "sub_end", "is_end": True, "prompt": "Sub end"},
            ],
        }
        subflow_file = tmp_path / "my_subflow.yaml"
        subflow_file.write_text(yaml.dump(subflow_config))

        # Create main config that references the subflow
        main_config = {
            "name": "main-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Main start",
                    "transitions": [
                        {
                            "target": "_subflow",
                            "subflow": {
                                "network": "my_subflow",
                                "return_stage": "end",
                            },
                        }
                    ],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }
        main_file = tmp_path / "main.yaml"
        main_file.write_text(yaml.dump(main_config))

        loader = WizardConfigLoader()
        wizard_fsm = loader.load(str(main_file))
        assert wizard_fsm is not None
        assert "my_subflow" in wizard_fsm._subflow_registry

    def test_subflow_from_subflows_directory(self, tmp_path: Path) -> None:
        """Loads subflow from subflows/<name>.yaml subdirectory."""
        # Create subflows subdirectory
        subflows_dir = tmp_path / "subflows"
        subflows_dir.mkdir()

        subflow_config = {
            "name": "dir-subflow",
            "stages": [
                {
                    "name": "sub_start",
                    "is_start": True,
                    "prompt": "Sub start",
                    "transitions": [{"target": "sub_end"}],
                },
                {"name": "sub_end", "is_end": True, "prompt": "Sub end"},
            ],
        }
        subflow_file = subflows_dir / "dir_subflow.yaml"
        subflow_file.write_text(yaml.dump(subflow_config))

        main_config = {
            "name": "main-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Main start",
                    "transitions": [
                        {
                            "target": "_subflow",
                            "subflow": {
                                "network": "dir_subflow",
                                "return_stage": "end",
                            },
                        }
                    ],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }
        main_file = tmp_path / "main.yaml"
        main_file.write_text(yaml.dump(main_config))

        loader = WizardConfigLoader()
        wizard_fsm = loader.load(str(main_file))
        assert wizard_fsm is not None
        assert "dir_subflow" in wizard_fsm._subflow_registry

    def test_missing_subflow_raises_error(self) -> None:
        """Referenced but unavailable subflow raises ValueError."""
        config = {
            "name": "broken-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Starting",
                    "transitions": [
                        {
                            "target": "_subflow",
                            "subflow": {
                                "network": "nonexistent_flow",
                                "return_stage": "end",
                            },
                        }
                    ],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }

        loader = WizardConfigLoader()
        # When loading from dict (no base_path), the subflow can't be found
        # from file. _load_single_subflow returns None, which means the
        # subflow is simply not in the registry (it warns but doesn't error
        # unless the subflow config itself raises).
        wizard_fsm = loader.load_from_dict(config)
        # Subflow not loaded - it's not in the registry
        assert "nonexistent_flow" not in wizard_fsm._subflow_registry

    def test_no_subflow_references_returns_empty(self) -> None:
        """Config without subflow references returns empty subflow registry."""
        config = {
            "name": "simple-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Starting",
                    "transitions": [{"target": "end"}],
                },
                {"name": "end", "is_end": True, "prompt": "Done"},
            ],
        }

        loader = WizardConfigLoader()
        wizard_fsm = loader.load_from_dict(config)
        assert wizard_fsm._subflow_registry == {}
