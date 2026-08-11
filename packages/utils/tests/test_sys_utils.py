"""Behavioural tests for :mod:`dataknobs_utils.sys_utils`.

``load_project_vars`` had none, which is how it came to promise
``Dict[str, str]`` while handing back a mapping that could contain ``None`` —
and to raise ``TypeError`` out of ``set_environ=True`` when it did.
"""

import os

import pytest

from dataknobs_utils import sys_utils


@pytest.fixture
def project_dir(tmp_path):
    """A directory holding a ``.project_vars`` with a valueless declaration.

    ``python-dotenv`` distinguishes the three spellings: ``A=1`` is a value,
    ``B=`` is the empty string, and a bare ``C`` is *no value at all*, which it
    reports as ``None``.
    """
    (tmp_path / ".project_vars").write_text("A=1\nB=\nC\n", encoding="utf-8")
    return tmp_path


def test_valueless_declaration_is_dropped(project_dir):
    """A key with no value does not reach the caller as ``None``."""
    config = sys_utils.load_project_vars(include_dot_env=False, start_path=project_dir)

    assert config == {"A": "1", "B": ""}


def test_set_environ_survives_a_valueless_declaration(project_dir, monkeypatch):
    """``set_environ=True`` must not raise on a valueless declaration.

    ``os.environ`` accepts only strings, so assigning the ``None`` such a line
    produces raised ``TypeError`` — from a helper whose whole purpose is to
    populate the environment.
    """
    monkeypatch.delenv("A", raising=False)
    monkeypatch.delenv("B", raising=False)
    monkeypatch.delenv("C", raising=False)

    sys_utils.load_project_vars(include_dot_env=False, set_environ=True, start_path=project_dir)

    assert os.environ["A"] == "1"
    assert os.environ["B"] == ""
    assert "C" not in os.environ


def test_existing_environment_wins(project_dir, monkeypatch):
    """A variable already in the environment is not overwritten."""
    monkeypatch.setenv("A", "already set")

    sys_utils.load_project_vars(include_dot_env=False, set_environ=True, start_path=project_dir)

    assert os.environ["A"] == "already set"


def test_missing_file_returns_none(tmp_path):
    """No project-vars file anywhere up the tree means ``None``."""
    assert sys_utils.load_project_vars(start_path=tmp_path, pvname=".no_such_file") is None


def test_dot_env_overrides_project_vars(project_dir):
    """``.env`` values win over ``.project_vars`` ones, as documented."""
    (project_dir / ".env").write_text("A=from-dot-env\n", encoding="utf-8")

    config = sys_utils.load_project_vars(start_path=project_dir)

    assert config is not None
    assert config["A"] == "from-dot-env"
    assert config["B"] == ""
