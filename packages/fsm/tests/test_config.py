"""Tests for FSM configuration modules."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
from pydantic import ValidationError

from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.loader import ConfigLoader
from dataknobs_fsm.config.schema import (
    ArcConfig,
    DataModeConfig,
    FSMConfig,
    NetworkConfig,
    ResourceConfig,
    StateConfig,
    UseCaseTemplate,
    apply_template,
    generate_json_schema,
    validate_config,
)
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.core.transactions import TransactionStrategy


class TestConfigSchema:
    """Test configuration schema definitions."""

    def test_state_config(self):
        """Test StateConfig creation and validation."""
        config = StateConfig(
            name="test_state",
            is_start=True,
            validators=[],
            transforms=[],
            arcs=[
                ArcConfig(target="next_state"),
            ],
        )
        
        assert config.name == "test_state"
        assert config.is_start is True
        assert len(config.arcs) == 1
        assert config.arcs[0].target == "next_state"

    def test_network_config(self):
        """Test NetworkConfig creation and validation."""
        config = NetworkConfig(
            name="test_network",
            states=[
                StateConfig(name="start", is_start=True, arcs=[]),
                StateConfig(name="end", is_end=True, arcs=[]),
            ],
        )
        
        assert config.name == "test_network"
        assert len(config.states) == 2
        assert config.states[0].is_start is True
        assert config.states[1].is_end is True

    def test_network_validation_no_start_state(self):
        """Test that network validation fails without start state."""
        with pytest.raises(ValidationError, match="at least one start state"):
            NetworkConfig(
                name="test_network",
                states=[
                    StateConfig(name="state1", arcs=[]),
                    StateConfig(name="state2", arcs=[]),
                ],
            )

    def test_network_validation_invalid_arc_target(self):
        """Test that network validation fails with invalid arc target."""
        with pytest.raises(ValidationError, match="not found in network"):
            NetworkConfig(
                name="test_network",
                states=[
                    StateConfig(
                        name="start",
                        is_start=True,
                        arcs=[ArcConfig(target="nonexistent")],
                    ),
                ],
            )

    def test_fsm_config(self):
        """Test FSMConfig creation and validation."""
        config = FSMConfig(
            name="test_fsm",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(name="start", is_start=True, arcs=[]),
                    ],
                ),
            ],
            main_network="main",
        )
        
        assert config.name == "test_fsm"
        assert config.main_network == "main"
        assert len(config.networks) == 1

    def test_fsm_validation_invalid_main_network(self):
        """Test that FSM validation fails with invalid main network."""
        with pytest.raises(ValidationError, match="Main network.*not found"):
            FSMConfig(
                name="test_fsm",
                networks=[
                    NetworkConfig(
                        name="network1",
                        states=[StateConfig(name="start", is_start=True, arcs=[])],
                    ),
                ],
                main_network="nonexistent",
            )

    def test_data_mode_config(self):
        """Test DataModeConfig."""
        config = DataModeConfig(
            default=DataHandlingMode.REFERENCE,
            state_overrides={"state1": DataHandlingMode.COPY},
        )
        
        assert config.default == DataHandlingMode.REFERENCE
        assert config.state_overrides["state1"] == DataHandlingMode.COPY

    def test_generate_json_schema(self):
        """Test JSON schema generation."""
        schema = generate_json_schema()
        
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "networks" in schema["properties"]
        assert "main_network" in schema["properties"]

    def test_validate_config_valid(self):
        """Test validate_config with valid configuration."""
        config_dict = {
            "name": "test",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {"name": "start", "is_start": True, "arcs": []},
                    ],
                },
            ],
            "main_network": "main",
        }
        
        config = validate_config(config_dict)
        assert isinstance(config, FSMConfig)
        assert config.name == "test"

    def test_validate_config_invalid(self):
        """Test validate_config with invalid configuration."""
        config_dict = {
            "name": "test",
            # Missing required fields
        }
        
        with pytest.raises(ValidationError):
            validate_config(config_dict)


class TestConfigTemplates:
    """Test configuration templates."""

    def test_apply_template_database_etl(self):
        """Test applying database ETL template."""
        config = apply_template(UseCaseTemplate.DATABASE_ETL)
        
        assert config["data_mode"]["default"] == DataHandlingMode.COPY
        assert config["transaction"]["strategy"] == TransactionStrategy.BATCH
        assert config["transaction"]["batch_size"] == 1000

    def test_apply_template_with_overrides(self):
        """Test applying template with overrides."""
        config = apply_template(
            UseCaseTemplate.FILE_PROCESSING,
            overrides={
                "data_mode": {"default": DataHandlingMode.COPY},
                "transaction": {"batch_size": 500},
            },
        )
        
        assert config["data_mode"]["default"] == DataHandlingMode.COPY
        assert config["transaction"]["batch_size"] == 500

    def test_all_templates(self):
        """Test that all templates are valid."""
        for template in UseCaseTemplate:
            config = apply_template(template)
            assert "data_mode" in config
            assert "transaction" in config
            assert "execution_strategy" in config


class TestConfigLoader:
    """Test configuration loader."""

    def test_load_from_dict(self):
        """Test loading configuration from dictionary."""
        loader = ConfigLoader()
        
        config_dict = {
            "name": "test",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {"name": "start", "is_start": True, "arcs": []},
                    ],
                },
            ],
            "main_network": "main",
        }
        
        config = loader.load_from_dict(config_dict)
        assert isinstance(config, FSMConfig)
        assert config.name == "test"

    def test_load_from_json_file(self, tmp_path):
        """Test loading configuration from JSON file."""
        config_dict = {
            "name": "test",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {"name": "start", "is_start": True, "arcs": []},
                    ],
                },
            ],
            "main_network": "main",
        }
        
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_dict, f)
        
        loader = ConfigLoader()
        config = loader.load_from_file(config_file)
        
        assert isinstance(config, FSMConfig)
        assert config.name == "test"

    def test_load_from_yaml_file(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_dict = {
            "name": "test",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {"name": "start", "is_start": True, "arcs": []},
                    ],
                },
            ],
            "main_network": "main",
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)
        
        loader = ConfigLoader()
        config = loader.load_from_file(config_file)
        
        assert isinstance(config, FSMConfig)
        assert config.name == "test"

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        loader = ConfigLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_from_file("nonexistent.json")

    def test_resolve_environment_variables(self, monkeypatch):
        """Test environment variable resolution."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        monkeypatch.setenv("FSM_PORT", "5432")
        
        loader = ConfigLoader()
        
        config_dict = {
            "name": "${TEST_VAR}",
            "metadata": {
                "port": "${FSM_PORT}",
                "optional": "${OPTIONAL_VAR:-default}",
            },
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {"name": "start", "is_start": True, "arcs": []},
                    ],
                },
            ],
            "main_network": "main",
        }
        
        config = loader.load_from_dict(config_dict)
        assert config.name == "test_value"
        assert config.metadata["port"] == "5432"
        assert config.metadata["optional"] == "default"

    def test_load_from_template(self):
        """Test loading configuration from template."""
        loader = ConfigLoader()
        
        # Need to provide full configuration when using template
        config = loader.load_from_template(
            UseCaseTemplate.DATABASE_ETL,
            overrides={
                "name": "test",
                "networks": [
                    {
                        "name": "main",
                        "states": [
                            {"name": "start", "is_start": True, "arcs": []},
                        ],
                    },
                ],
                "main_network": "main",
            },
        )
        
        assert isinstance(config, FSMConfig)
        assert config.data_mode.default == DataHandlingMode.COPY

    def test_merge_configs(self):
        """Test merging multiple configurations."""
        loader = ConfigLoader()
        
        config1 = FSMConfig(
            name="test1",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[StateConfig(name="start", is_start=True, arcs=[])],
                ),
            ],
            main_network="main",
        )
        
        config2 = FSMConfig(
            name="test2",
            version="2.0.0",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[StateConfig(name="start", is_start=True, arcs=[])],
                ),
            ],
            main_network="main",
        )
        
        merged = loader.merge_configs(config1, config2)
        assert merged.name == "test2"
        assert merged.version == "2.0.0"


class TestFSMBuilder:
    """Test FSM builder."""

    def test_build_simple_fsm(self):
        """Test building a simple FSM."""
        config = FSMConfig(
            name="test",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(name="start", is_start=True, arcs=[]),
                        StateConfig(name="end", is_end=True, arcs=[]),
                    ],
                ),
            ],
            main_network="main",
        )
        
        builder = FSMBuilder()
        fsm = builder.build(config)
        
        assert fsm.config == config
        assert "main" in fsm.networks
        assert fsm.main_network.name == "main"

    def test_register_custom_function(self):
        """Test registering custom functions."""
        def custom_validator(data):
            return "test" in data
        
        builder = FSMBuilder()
        builder.register_function("custom_validator", custom_validator)
        
        assert builder._function_manager.has_function("custom_validator")

    def test_build_with_resources(self):
        """Test building FSM with resources."""
        config = FSMConfig(
            name="test",
            resources=[
                ResourceConfig(
                    name="test_db",
                    type="database",
                    config={"connection_string": "sqlite:///:memory:"},
                ),
            ],
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(
                            name="start",
                            is_start=True,
                            resources=["test_db"],
                            arcs=[],
                        ),
                    ],
                ),
            ],
            main_network="main",
        )
        
        builder = FSMBuilder()
        # Test that building with valid resources works
        fsm = builder.build(config)
        assert fsm is not None
        assert fsm.name == "test"
        
        # Test that custom resource without class raises error
        invalid_config = FSMConfig(
            name="test",
            resources=[
                ResourceConfig(
                    name="invalid_resource",
                    type="custom",  # Custom type requires 'class' in config
                    config={},  # Missing 'class' should cause error
                ),
            ],
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(
                            name="start",
                            is_start=True,
                            arcs=[],
                        ),
                    ],
                ),
            ],
            main_network="main",
        )
        
        with pytest.raises(ValueError, match="Custom resource requires 'class'"):
            builder.build(invalid_config)

    def test_validate_completeness(self):
        """Test FSM completeness validation."""
        # Invalid arc target
        config = FSMConfig(
            name="test",
            networks=[
                NetworkConfig(
                    name="main",
                    states=[
                        StateConfig(
                            name="start",
                            is_start=True,
                            arcs=[],  # This is valid - no arcs
                        ),
                    ],
                ),
            ],
            main_network="main",
        )
        
        builder = FSMBuilder()
        fsm = builder.build(config)  # Should not raise
        assert fsm.validate()


@pytest.fixture
def sample_config_dict():
    """Provide a sample configuration dictionary."""
    return {
        "name": "sample_fsm",
        "version": "1.0.0",
        "description": "Sample FSM for testing",
        "data_mode": {
            "default": "copy",
            "state_overrides": {
                "process": "reference",
            },
        },
        "transaction": {
            "strategy": "batch",
            "batch_size": 100,
        },
        "resources": [
            {
                "name": "main_db",
                "type": "database",
                "config": {
                    "host": "localhost",
                    "port": 5432,
                },
            },
        ],
        "networks": [
            {
                "name": "main",
                "states": [
                    {
                        "name": "start",
                        "is_start": True,
                        "arcs": [
                            {"target": "process"},
                        ],
                    },
                    {
                        "name": "process",
                        "arcs": [
                            {"target": "end"},
                        ],
                    },
                    {
                        "name": "end",
                        "is_end": True,
                        "arcs": [],
                    },
                ],
            },
        ],
        "main_network": "main",
    }


def test_full_config_workflow(sample_config_dict, tmp_path):
    """Test complete configuration workflow."""
    # Save to file
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config_dict, f)
    
    # Load from file
    loader = ConfigLoader()
    config = loader.load_from_file(config_file)
    
    # Validate
    assert config.name == "sample_fsm"
    assert config.data_mode.default == DataHandlingMode.COPY
    assert len(config.networks) == 1
    assert len(config.resources) == 1
    
    # Build FSM
    builder = FSMBuilder()
    fsm = builder.build(config)
    
    # Verify FSM was built correctly
    assert fsm is not None
    assert fsm.config == config
    assert len(fsm.networks) == 1
    assert "main" in fsm.networks
