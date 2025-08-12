"""Pytest configuration and fixtures for config package tests."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        "database": [
            {
                "name": "primary",
                "host": "localhost",
                "port": 5432,
                "database": "myapp",
            },
            {
                "name": "secondary",
                "host": "backup.example.com",
                "port": 5432,
                "database": "myapp_backup",
            },
        ],
        "cache": [
            {
                "name": "redis",
                "host": "localhost",
                "port": 6379,
                "ttl": 3600,
            }
        ],
        "settings": {
            "config_root": "/app/config",
            "global_root": "/app",
            "database.global_root": "/app/db",
            "path_resolution_attributes": ["config_path", "database.data_dir"],
            "default_timeout": 30,
            "database.default_pool_size": 10,
        },
    }


@pytest.fixture
def env_vars(monkeypatch):
    """Helper to set environment variables."""

    def _set_env(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, str(value))

    return _set_env


@pytest.fixture
def clear_env(monkeypatch):
    """Clear all DATAKNOBS_ environment variables."""
    for key in list(os.environ.keys()):
        if key.startswith("DATAKNOBS_"):
            monkeypatch.delenv(key)
