"""Tests for configuration validator module."""

import json
import tempfile
from pathlib import Path
import pytest
import yaml

from dataknobs_fsm.config.validator import ConfigValidator
from dataknobs_fsm.config.schema import FSMConfig


class TestConfigValidator:
    """Test suite for ConfigValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a ConfigValidator instance."""
        return ConfigValidator()

    @pytest.fixture
    def valid_config_dict(self):
        """Create a valid configuration dictionary."""
        return {
            "name": "test_fsm",
            "version": "1.0.0",
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {
                            "name": "start",
                            "is_start": True,
                            "arcs": [{"target": "end"}]
                        },
                        {
                            "name": "end",
                            "is_end": True,
                            "arcs": []
                        }
                    ]
                }
            ],
            "main_network": "main"
        }

    @pytest.fixture
    def invalid_config_dict(self):
        """Create an invalid configuration dictionary."""
        return {
            "name": "test_fsm"
            # Missing required fields: networks, main_network
        }

    def test_validate_dict_with_valid_config(self, validator, valid_config_dict):
        """Test validating a valid configuration dictionary."""
        errors = validator.validate_dict(valid_config_dict)
        assert errors == []

    def test_validate_dict_with_invalid_config(self, validator, invalid_config_dict):
        """Test validating an invalid configuration dictionary."""
        errors = validator.validate_dict(invalid_config_dict)
        assert len(errors) > 0
        # Check that error message contains useful information
        assert any("validation error" in err.lower() or "missing" in err.lower()
                  for err in errors)

    def test_validate_dict_with_empty_config(self, validator):
        """Test validating an empty configuration dictionary."""
        errors = validator.validate_dict({})
        assert len(errors) > 0

    def test_validate_dict_with_none(self, validator):
        """Test validating None as configuration."""
        errors = validator.validate_dict(None)
        assert len(errors) > 0

    def test_validate_file_with_valid_json(self, validator, valid_config_dict):
        """Test validating a valid JSON configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_config_dict, f)
            temp_path = f.name

        try:
            errors = validator.validate_file(temp_path)
            assert errors == []
        finally:
            Path(temp_path).unlink()

    def test_validate_file_with_valid_yaml(self, validator, valid_config_dict):
        """Test validating a valid YAML configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config_dict, f)
            temp_path = f.name

        try:
            errors = validator.validate_file(temp_path)
            assert errors == []
        finally:
            Path(temp_path).unlink()

    def test_validate_file_with_invalid_json(self, validator, invalid_config_dict):
        """Test validating an invalid JSON configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config_dict, f)
            temp_path = f.name

        try:
            errors = validator.validate_file(temp_path)
            assert len(errors) > 0
        finally:
            Path(temp_path).unlink()

    def test_validate_file_with_malformed_json(self, validator):
        """Test validating a malformed JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ this is not valid json }")
            temp_path = f.name

        try:
            errors = validator.validate_file(temp_path)
            assert len(errors) > 0
            # Should contain some error message about the invalid format
            # The exact message may vary, so just check that an error was caught
            assert errors[0]  # At least one error message exists
        finally:
            Path(temp_path).unlink()

    def test_validate_file_nonexistent(self, validator):
        """Test validating a non-existent file."""
        errors = validator.validate_file("/path/that/does/not/exist.json")
        assert len(errors) > 0
        # Should contain file not found error
        assert any("not found" in err.lower() or "exist" in err.lower()
                  for err in errors)

    def test_validate_file_with_empty_file(self, validator):
        """Test validating an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Write nothing to the file
            temp_path = f.name

        try:
            errors = validator.validate_file(temp_path)
            assert len(errors) > 0
        finally:
            Path(temp_path).unlink()

    def test_validate_dict_with_complex_valid_config(self, validator):
        """Test validating a complex but valid configuration."""
        complex_config = {
            "name": "complex_fsm",
            "version": "1.0.0",
            "description": "A complex FSM for testing",
            "resources": [
                {
                    "name": "db1",
                    "type": "database",
                    "config": {
                        "connection_string": "sqlite:///test.db"
                    }
                }
            ],
            "networks": [
                {
                    "name": "main",
                    "states": [
                        {
                            "name": "init",
                            "is_start": True,
                            "resources": ["db1"],
                            "arcs": [
                                {
                                    "target": "process",
                                    "condition": {
                                        "type": "inline",
                                        "code": "lambda data: data.get('ready', False)"
                                    }
                                }
                            ]
                        },
                        {
                            "name": "process",
                            "arcs": [{"target": "done"}]
                        },
                        {
                            "name": "done",
                            "is_end": True,
                            "arcs": []
                        }
                    ],
                    "resources": ["db1"]
                }
            ],
            "main_network": "main"
        }

        errors = validator.validate_dict(complex_config)
        assert errors == []

    def test_validate_dict_handles_exception_gracefully(self, validator, monkeypatch):
        """Test that validation handles unexpected exceptions gracefully."""
        def mock_validate_config(config):
            raise RuntimeError("Unexpected error during validation")

        # Mock the validate_config function to raise an exception
        monkeypatch.setattr(
            "dataknobs_fsm.config.validator.validate_config",
            mock_validate_config
        )

        errors = validator.validate_dict({"fsm": {}})
        assert len(errors) == 1
        assert "Unexpected error during validation" in errors[0]

    def test_validate_file_handles_exception_gracefully(self, validator, monkeypatch):
        """Test that file validation handles unexpected exceptions gracefully."""
        def mock_load_file(self, path):
            raise RuntimeError("Unexpected error during file loading")

        # Mock the loader's load_file method to raise an exception
        monkeypatch.setattr(
            "dataknobs_fsm.config.loader.ConfigLoader.load_from_file",
            mock_load_file
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"fsm": {}}, f)
            temp_path = f.name

        try:
            errors = validator.validate_file(temp_path)
            assert len(errors) == 1
            assert "Unexpected error during file loading" in errors[0]
        finally:
            Path(temp_path).unlink()