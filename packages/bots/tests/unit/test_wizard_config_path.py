"""Tests for wizard config path resolution (Item 20).

Verifies that relative ``wizard_config`` paths in inline bot configs are
resolved against ``config_base_path`` when provided.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dataknobs_bots.reasoning.wizard import WizardReasoning


def _minimal_wizard_yaml() -> dict:
    """Return a minimal valid wizard config dict."""
    return {
        "name": "test-wizard",
        "stages": [
            {
                "name": "start",
                "is_start": True,
                "is_end": True,
                "prompt": "Hi",
            },
        ],
    }


class TestWizardConfigPathResolution:
    """Tests for config_base_path support in WizardReasoning.from_config()."""

    def test_relative_path_resolved_against_config_base_path(
        self, tmp_path: Path
    ) -> None:
        """Relative wizard_config path should resolve against config_base_path."""
        wizard_dir = tmp_path / "wizards"
        wizard_dir.mkdir()
        wizard_file = wizard_dir / "flow.yaml"
        wizard_file.write_text(yaml.dump(_minimal_wizard_yaml()))

        config = {
            "strategy": "wizard",
            "wizard_config": "wizards/flow.yaml",
            "config_base_path": str(tmp_path),
        }
        reasoning = WizardReasoning.from_config(config)
        assert reasoning is not None
        assert reasoning._wizard_fsm is not None

    def test_absolute_path_ignores_config_base_path(
        self, tmp_path: Path
    ) -> None:
        """Absolute wizard_config paths should work regardless of config_base_path."""
        wizard_file = tmp_path / "flow.yaml"
        wizard_file.write_text(yaml.dump(_minimal_wizard_yaml()))

        config = {
            "strategy": "wizard",
            "wizard_config": str(wizard_file),
            "config_base_path": "/nonexistent/path",
        }
        reasoning = WizardReasoning.from_config(config)
        assert reasoning is not None

    def test_relative_path_without_base_path_uses_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without config_base_path, relative paths resolve against CWD (backward compat)."""
        wizard_dir = tmp_path / "wizards"
        wizard_dir.mkdir()
        wizard_file = wizard_dir / "flow.yaml"
        wizard_file.write_text(yaml.dump(_minimal_wizard_yaml()))

        monkeypatch.chdir(tmp_path)

        config = {
            "strategy": "wizard",
            "wizard_config": "wizards/flow.yaml",
        }
        reasoning = WizardReasoning.from_config(config)
        assert reasoning is not None

    def test_inline_dict_ignores_config_base_path(self) -> None:
        """Inline dict wizard_config should be unaffected by config_base_path."""
        config = {
            "strategy": "wizard",
            "wizard_config": _minimal_wizard_yaml(),
            "config_base_path": "/nonexistent/path",
        }
        reasoning = WizardReasoning.from_config(config)
        assert reasoning is not None

    def test_base_path_propagated_to_subflow_resolution(
        self, tmp_path: Path
    ) -> None:
        """config_base_path should be passed through to loader for subflow resolution."""
        # Create main wizard with a subflow reference
        subflow_dir = tmp_path / "subflows"
        subflow_dir.mkdir()
        subflow_file = subflow_dir / "sub.yaml"
        subflow_file.write_text(
            yaml.dump(
                {
                    "name": "sub-wizard",
                    "stages": [
                        {
                            "name": "sub_start",
                            "is_start": True,
                            "is_end": True,
                            "prompt": "Sub",
                        },
                    ],
                }
            )
        )

        main_wizard = {
            "name": "main-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "prompt": "Hello",
                    "transitions": [
                        {
                            "target": "end",
                            "subflow": {
                                "wizard_config": "subflows/sub.yaml",
                            },
                        }
                    ],
                },
                {
                    "name": "end",
                    "is_end": True,
                    "prompt": "Done",
                },
            ],
        }
        main_file = tmp_path / "main.yaml"
        main_file.write_text(yaml.dump(main_wizard))

        config = {
            "strategy": "wizard",
            "wizard_config": "main.yaml",
            "config_base_path": str(tmp_path),
        }
        reasoning = WizardReasoning.from_config(config)
        assert reasoning is not None


class TestBotConfigBasePathPropagation:
    """Tests for config_base_path propagation from bot config to reasoning."""

    @pytest.mark.asyncio
    async def test_bot_config_base_path_propagated_to_reasoning(
        self, tmp_path: Path
    ) -> None:
        """config_base_path at bot level should propagate to reasoning config.

        Exercises the full DynaBot.from_config() path to verify that
        _build_from_config propagates config_base_path into the reasoning
        config dict before calling create_reasoning_from_config.
        """
        from dataknobs_bots.bot.base import DynaBot

        wizard_dir = tmp_path / "wizards"
        wizard_dir.mkdir()
        wizard_file = wizard_dir / "flow.yaml"
        wizard_file.write_text(yaml.dump(_minimal_wizard_yaml()))

        bot_config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "wizard",
                "wizard_config": "wizards/flow.yaml",
            },
            "config_base_path": str(tmp_path),
        }

        bot = await DynaBot.from_config(bot_config)
        try:
            assert bot is not None
            assert bot.reasoning_strategy is not None
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_reasoning_level_base_path_takes_precedence(
        self, tmp_path: Path
    ) -> None:
        """Reasoning-level config_base_path wins over bot-level."""
        from dataknobs_bots.bot.base import DynaBot

        wizard_dir = tmp_path / "wizards"
        wizard_dir.mkdir()
        wizard_file = wizard_dir / "flow.yaml"
        wizard_file.write_text(yaml.dump(_minimal_wizard_yaml()))

        bot_config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "wizard",
                "wizard_config": "wizards/flow.yaml",
                "config_base_path": str(tmp_path),  # reasoning-level
            },
            "config_base_path": "/nonexistent/should/be/ignored",
        }

        bot = await DynaBot.from_config(bot_config)
        try:
            assert bot is not None
            assert bot.reasoning_strategy is not None
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_conflicting_base_paths_logs_debug(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When both levels specify different config_base_path, log a debug message."""
        import logging

        from dataknobs_bots.bot.base import DynaBot

        wizard_dir = tmp_path / "wizards"
        wizard_dir.mkdir()
        wizard_file = wizard_dir / "flow.yaml"
        wizard_file.write_text(yaml.dump(_minimal_wizard_yaml()))

        bot_config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "wizard",
                "wizard_config": "wizards/flow.yaml",
                "config_base_path": str(tmp_path),
            },
            "config_base_path": "/different/path",
        }

        with caplog.at_level(logging.DEBUG, logger="dataknobs_bots.bot.base"):
            bot = await DynaBot.from_config(bot_config)
            try:
                assert bot is not None
            finally:
                await bot.close()

        assert any(
            "config_base_path" in record.message and "ignoring" in record.message.lower()
            for record in caplog.records
        )


class TestDynaBotConfigBuilderBasePath:
    """Tests for DynaBotConfigBuilder.set_config_base_path()."""

    def test_set_config_base_path(self) -> None:
        """set_config_base_path should add config_base_path to bot config."""
        from dataknobs_bots.config.builder import DynaBotConfigBuilder

        builder = (
            DynaBotConfigBuilder()
            .set_llm("echo", model="test")
            .set_conversation_storage("memory")
        )
        result = builder.set_config_base_path("/path/to/configs")
        assert result is builder  # fluent API
        config = builder.build()
        assert config["config_base_path"] == "/path/to/configs"

    def test_set_config_base_path_with_path_object(self) -> None:
        """set_config_base_path should accept Path objects."""
        from dataknobs_bots.config.builder import DynaBotConfigBuilder

        builder = (
            DynaBotConfigBuilder()
            .set_llm("echo", model="test")
            .set_conversation_storage("memory")
        )
        builder.set_config_base_path(Path("/path/to/configs"))
        config = builder.build()
        assert config["config_base_path"] == "/path/to/configs"
