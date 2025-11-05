"""Unit tests for tool resolution from configuration."""

import pytest

from dataknobs_bots import DynaBot
from tests.fixtures.test_tools import SimpleTestTool, ParameterizedTestTool


class TestToolResolution:
    """Test tool resolution from configuration."""

    def test_resolve_simple_tool_by_class(self):
        """Test resolving a simple tool by class path."""
        tool_config = {
            "class": "tests.fixtures.test_tools.SimpleTestTool",
            "params": {},
        }

        config = {}

        tool = DynaBot._resolve_tool(tool_config, config)

        assert tool is not None
        assert isinstance(tool, SimpleTestTool)
        assert tool.name == "simple_test"

    def test_resolve_parameterized_tool(self):
        """Test resolving a tool with parameters."""
        tool_config = {
            "class": "tests.fixtures.test_tools.ParameterizedTestTool",
            "params": {
                "prefix": "custom",
                "multiplier": 3,
            },
        }

        config = {}

        tool = DynaBot._resolve_tool(tool_config, config)

        assert tool is not None
        assert isinstance(tool, ParameterizedTestTool)
        assert tool.prefix == "custom"
        assert tool.multiplier == 3

    def test_resolve_tool_with_xref(self):
        """Test resolving a tool using xref."""
        config = {
            "tool_definitions": {
                "my_tool": {
                    "class": "tests.fixtures.test_tools.SimpleTestTool",
                    "params": {},
                }
            }
        }

        tool_config = "xref:tools[my_tool]"

        tool = DynaBot._resolve_tool(tool_config, config)

        assert tool is not None
        assert isinstance(tool, SimpleTestTool)

    def test_resolve_tool_with_xref_dict_format(self):
        """Test resolving a tool using xref in dict format."""
        config = {
            "tool_definitions": {
                "my_tool": {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {"prefix": "xref"},
                }
            }
        }

        tool_config = {"xref": "xref:tools[my_tool]"}

        tool = DynaBot._resolve_tool(tool_config, config)

        assert tool is not None
        assert isinstance(tool, ParameterizedTestTool)
        assert tool.prefix == "xref"

    def test_resolve_tool_with_nested_xref(self):
        """Test resolving a tool with nested xref references."""
        config = {
            "tool_definitions": {
                "base_tool": {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {"prefix": "base", "multiplier": 2},
                },
                "alias_tool": {"xref": "xref:tools[base_tool]"},
            }
        }

        tool_config = "xref:tools[alias_tool]"

        tool = DynaBot._resolve_tool(tool_config, config)

        assert tool is not None
        assert isinstance(tool, ParameterizedTestTool)
        assert tool.prefix == "base"
        assert tool.multiplier == 2

    def test_resolve_invalid_class_path(self):
        """Test handling of invalid class path."""
        tool_config = {
            "class": "non.existent.Module",
            "params": {},
        }

        config = {}

        tool = DynaBot._resolve_tool(tool_config, config)

        # Should return None for invalid class
        assert tool is None

    def test_resolve_invalid_xref(self):
        """Test handling of invalid xref."""
        config = {
            "tool_definitions": {
                "existing_tool": {
                    "class": "tests.fixtures.test_tools.SimpleTestTool",
                }
            }
        }

        tool_config = "xref:tools[non_existent]"

        tool = DynaBot._resolve_tool(tool_config, config)

        # Should return None for non-existent tool
        assert tool is None

    def test_resolve_invalid_xref_format(self):
        """Test handling of invalid xref format."""
        config = {}

        tool_config = "xref:invalid_format"

        tool = DynaBot._resolve_tool(tool_config, config)

        # Should return None for invalid format
        assert tool is None

    def test_resolve_non_tool_class(self):
        """Test handling of class that's not a Tool."""
        tool_config = {
            "class": "builtins.str",  # Not a Tool subclass
            "params": {},
        }

        config = {}

        tool = DynaBot._resolve_tool(tool_config, config)

        # Should return None for non-Tool class
        assert tool is None

    def test_resolve_missing_class_key(self):
        """Test handling of config without class or xref key."""
        tool_config = {
            "name": "something",
            "other": "data",
        }

        config = {}

        tool = DynaBot._resolve_tool(tool_config, config)

        # Should return None for invalid config
        assert tool is None

    @pytest.mark.asyncio
    async def test_from_config_with_tool_resolution(self):
        """Test complete bot creation with tool resolution."""
        config = {
            "llm": {
                "provider": "ollama",
                "model": "phi3:mini",
                "temperature": 0.7,
            },
            "conversation_storage": {
                "backend": "memory",
            },
            "tools": [
                {
                    "class": "tests.fixtures.test_tools.SimpleTestTool",
                    "params": {},
                },
                {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {"prefix": "config", "multiplier": 2},
                },
            ],
        }

        bot = await DynaBot.from_config(config)

        # Verify tools were registered
        tools = list(bot.tool_registry)
        assert len(tools) == 2

        # Verify tool types
        tool_types = {type(tool) for tool in tools}
        assert SimpleTestTool in tool_types
        assert ParameterizedTestTool in tool_types

        # Verify parameterized tool got correct params
        param_tool = next(t for t in tools if isinstance(t, ParameterizedTestTool))
        assert param_tool.prefix == "config"
        assert param_tool.multiplier == 2

    @pytest.mark.asyncio
    async def test_from_config_with_xref_tools(self):
        """Test bot creation with xref-based tool configuration."""
        config = {
            "llm": {
                "provider": "ollama",
                "model": "phi3:mini",
                "temperature": 0.7,
            },
            "conversation_storage": {
                "backend": "memory",
            },
            "tool_definitions": {
                "simple": {
                    "class": "tests.fixtures.test_tools.SimpleTestTool",
                    "params": {},
                },
                "complex": {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {"prefix": "xref-tool"},
                },
            },
            "tools": [
                "xref:tools[simple]",
                "xref:tools[complex]",
            ],
        }

        bot = await DynaBot.from_config(config)

        # Verify tools were registered
        tools = list(bot.tool_registry)
        assert len(tools) == 2

        # Verify xref resolved correctly
        param_tool = next(t for t in tools if isinstance(t, ParameterizedTestTool))
        assert param_tool.prefix == "xref-tool"

    @pytest.mark.asyncio
    async def test_from_config_skips_invalid_tools(self):
        """Test that bot creation continues when some tools fail to resolve."""
        config = {
            "llm": {
                "provider": "ollama",
                "model": "phi3:mini",
                "temperature": 0.7,
            },
            "conversation_storage": {
                "backend": "memory",
            },
            "tools": [
                {
                    "class": "tests.fixtures.test_tools.SimpleTestTool",
                    "params": {},
                },
                {
                    "class": "non.existent.Tool",  # This will fail
                    "params": {},
                },
                {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {},
                },
            ],
        }

        bot = await DynaBot.from_config(config)

        # Should have 2 tools (skipped the invalid one)
        tools = list(bot.tool_registry)
        assert len(tools) == 2

        # Verify correct tools were loaded
        tool_types = {type(tool) for tool in tools}
        assert SimpleTestTool in tool_types
        assert ParameterizedTestTool in tool_types

    @pytest.mark.asyncio
    async def test_tool_execution_after_resolution(self):
        """Test that resolved tools can be executed."""
        config = {
            "llm": {
                "provider": "ollama",
                "model": "phi3:mini",
                "temperature": 0.7,
            },
            "conversation_storage": {
                "backend": "memory",
            },
            "tools": [
                {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {"prefix": "exec", "multiplier": 2},
                }
            ],
        }

        bot = await DynaBot.from_config(config)

        # Get the tool
        tools = list(bot.tool_registry)
        assert len(tools) == 1
        tool = tools[0]

        # Execute the tool
        result = await tool.execute(value="test")
        assert result == "exec:testexec:test"  # multiplier=2
