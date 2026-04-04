"""Tests for sandboxed Jinja2 template environment.

Verifies that ``create_template_env`` returns a ``SandboxedEnvironment``
that blocks SSTI payloads while preserving normal template functionality.
"""

from __future__ import annotations

import pytest
from jinja2.exceptions import SecurityError, UndefinedError
from jinja2.sandbox import SandboxedEnvironment

from dataknobs_bots.utils.template_env import create_template_env


class TestCreateTemplateEnv:
    """Tests for the ``create_template_env`` factory."""

    def test_returns_sandboxed_environment(self) -> None:
        env = create_template_env()
        assert isinstance(env, SandboxedEnvironment)

    def test_strict_mode_returns_sandboxed_environment(self) -> None:
        env = create_template_env(strict=True)
        assert isinstance(env, SandboxedEnvironment)


class TestNormalRendering:
    """Verify that normal template functionality is preserved."""

    def test_simple_variable(self) -> None:
        env = create_template_env()
        result = env.from_string("Hello {{ name }}!").render(name="Alice")
        assert result == "Hello Alice!"

    def test_dict_get(self) -> None:
        env = create_template_env()
        result = env.from_string(
            "{{ data.get('goal', 'default') }}"
        ).render(data={"goal": "fitness"})
        assert result == "fitness"

    def test_dict_get_fallback(self) -> None:
        env = create_template_env()
        result = env.from_string(
            "{{ data.get('missing', 'fallback') }}"
        ).render(data={})
        assert result == "fallback"

    def test_loop(self) -> None:
        env = create_template_env()
        result = env.from_string(
            "{% for item in items %}{{ item }} {% endfor %}"
        ).render(items=["a", "b", "c"])
        assert result == "a b c "

    def test_conditional(self) -> None:
        env = create_template_env()
        result = env.from_string(
            "{% if active %}yes{% else %}no{% endif %}"
        ).render(active=True)
        assert result == "yes"

    def test_filter(self) -> None:
        env = create_template_env()
        result = env.from_string("{{ name | upper }}").render(name="alice")
        assert result == "ALICE"

    def test_undefined_variable_renders_empty(self) -> None:
        env = create_template_env()
        result = env.from_string("Hello {{ missing }}!").render()
        assert result == "Hello !"

    def test_strict_mode_raises_on_undefined(self) -> None:
        env = create_template_env(strict=True)
        with pytest.raises(UndefinedError):
            env.from_string("Hello {{ missing }}!").render()

    def test_strict_mode_renders_defined_vars(self) -> None:
        env = create_template_env(strict=True)
        result = env.from_string("Hello {{ name }}!").render(name="Bob")
        assert result == "Hello Bob!"


class TestSSTIBlocked:
    """Verify that server-side template injection payloads are blocked."""

    def test_mro_subclasses_chain(self) -> None:
        """The canonical SSTI RCE payload is blocked."""
        env = create_template_env()
        with pytest.raises(SecurityError):
            env.from_string(
                "{{ ''.__class__.__mro__[1].__subclasses__() }}"
            ).render()

    def test_globals_access_via_cycler(self) -> None:
        env = create_template_env()
        with pytest.raises(SecurityError):
            env.from_string(
                "{{ cycler.__init__.__globals__ }}"
            ).render()

    def test_globals_access_via_string(self) -> None:
        env = create_template_env()
        with pytest.raises(SecurityError):
            env.from_string(
                "{{ ''.__init__.__globals__ }}"
            ).render()

    def test_ssti_in_strict_mode(self) -> None:
        env = create_template_env(strict=True)
        with pytest.raises(SecurityError):
            env.from_string("{{ ''.__class__ }}").render()
