"""Integration tests for config-based tool loading.

These tests verify that:
1. Tools can be loaded via direct class instantiation from config
2. Tools can be loaded via xref references
3. Tool parameters are properly applied from config
4. Tools work correctly with ReAct reasoning

Most tests use the Echo LLM provider for fast, deterministic testing.

Run all tests:
    pytest tests/integration/test_config_tools_integration.py
"""

import os

import pytest

from dataknobs_bots import BotContext, DynaBot
from tests.fixtures.test_tools import SimpleTestTool, ParameterizedTestTool


# =============================================================================
# Tests using Echo LLM (fast, no external dependencies)
# =============================================================================


class TestDirectToolInstantiation:
    """Test direct tool instantiation from config using Echo LLM."""

    @pytest.mark.asyncio
    async def test_single_tool_from_config(self, echo_config):
        """Test loading a single tool via direct class instantiation."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "tools": [
                {
                    "class": "tests.fixtures.test_tools.SimpleTestTool",
                    "params": {},
                }
            ],
        }

        bot = await DynaBot.from_config(config)

        # Verify tool was loaded
        tools = list(bot.tool_registry)
        assert len(tools) == 1
        assert isinstance(tools[0], SimpleTestTool)

    @pytest.mark.asyncio
    async def test_tool_with_parameters(self, echo_config):
        """Test loading a tool with custom parameters."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "tools": [
                {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {"prefix": "custom", "multiplier": 3},
                }
            ],
        }

        bot = await DynaBot.from_config(config)

        # Verify tool was loaded with correct parameters
        tools = list(bot.tool_registry)
        assert len(tools) == 1
        tool = tools[0]
        assert isinstance(tool, ParameterizedTestTool)
        assert tool.prefix == "custom"
        assert tool.multiplier == 3

    @pytest.mark.asyncio
    async def test_multiple_tools_from_config(self, echo_config):
        """Test loading multiple tools from config."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "tools": [
                {
                    "class": "tests.fixtures.test_tools.SimpleTestTool",
                    "params": {},
                },
                {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {"prefix": "tool2"},
                },
            ],
        }

        bot = await DynaBot.from_config(config)

        # Verify both tools were loaded
        tools = list(bot.tool_registry)
        assert len(tools) == 2

        tool_types = {type(tool) for tool in tools}
        assert SimpleTestTool in tool_types
        assert ParameterizedTestTool in tool_types


class TestXRefToolLoading:
    """Test xref-based tool loading using Echo LLM."""

    @pytest.mark.asyncio
    async def test_xref_string_format(self, echo_config):
        """Test loading tool via xref string format."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "tool_definitions": {
                "my_tool": {
                    "class": "tests.fixtures.test_tools.SimpleTestTool",
                    "params": {},
                }
            },
            "tools": ["xref:tools[my_tool]"],
        }

        bot = await DynaBot.from_config(config)

        # Verify tool was loaded via xref
        tools = list(bot.tool_registry)
        assert len(tools) == 1
        assert isinstance(tools[0], SimpleTestTool)

    @pytest.mark.asyncio
    async def test_xref_with_parameters(self, echo_config):
        """Test xref tool with custom parameters."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "tool_definitions": {
                "parameterized": {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {"prefix": "xref", "multiplier": 5},
                }
            },
            "tools": ["xref:tools[parameterized]"],
        }

        bot = await DynaBot.from_config(config)

        # Verify tool parameters from xref
        tools = list(bot.tool_registry)
        assert len(tools) == 1
        tool = tools[0]
        assert isinstance(tool, ParameterizedTestTool)
        assert tool.prefix == "xref"
        assert tool.multiplier == 5

    @pytest.mark.asyncio
    async def test_multiple_xref_tools(self, echo_config):
        """Test loading multiple tools via xref."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "tool_definitions": {
                "tool1": {
                    "class": "tests.fixtures.test_tools.SimpleTestTool",
                    "params": {},
                },
                "tool2": {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {"prefix": "second"},
                },
            },
            "tools": ["xref:tools[tool1]", "xref:tools[tool2]"],
        }

        bot = await DynaBot.from_config(config)

        # Verify both tools loaded
        tools = list(bot.tool_registry)
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_nested_xref(self, echo_config):
        """Test nested xref references."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "tool_definitions": {
                "base": {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {"prefix": "base", "multiplier": 2},
                },
                "alias": {"xref": "xref:tools[base]"},
            },
            "tools": ["xref:tools[alias]"],
        }

        bot = await DynaBot.from_config(config)

        # Verify nested xref resolved correctly
        tools = list(bot.tool_registry)
        assert len(tools) == 1
        tool = tools[0]
        assert isinstance(tool, ParameterizedTestTool)
        assert tool.prefix == "base"
        assert tool.multiplier == 2


class TestMixedToolLoading:
    """Test mixing direct and xref-based tool loading using Echo LLM."""

    @pytest.mark.asyncio
    async def test_direct_and_xref_together(self, echo_config):
        """Test using both direct instantiation and xref in same config."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "tool_definitions": {
                "defined_tool": {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {"prefix": "xref"},
                }
            },
            "tools": [
                {
                    "class": "tests.fixtures.test_tools.SimpleTestTool",
                    "params": {},
                },
                "xref:tools[defined_tool]",
            ],
        }

        bot = await DynaBot.from_config(config)

        # Verify both loading methods work together
        tools = list(bot.tool_registry)
        assert len(tools) == 2

        tool_types = {type(tool) for tool in tools}
        assert SimpleTestTool in tool_types
        assert ParameterizedTestTool in tool_types

    @pytest.mark.asyncio
    async def test_reusable_tool_definitions(self, echo_config):
        """Test that tool definitions can be reused multiple times."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "tool_definitions": {
                "template": {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {"prefix": "shared", "multiplier": 1},
                }
            },
            "tools": [
                # Can't actually use same xref twice (would register same instance)
                # But we can reference it once to prove the pattern works
                "xref:tools[template]",
            ],
        }

        bot = await DynaBot.from_config(config)

        # Verify tool definition was used
        tools = list(bot.tool_registry)
        assert len(tools) == 1
        tool = tools[0]
        assert tool.prefix == "shared"


class TestToolExecutionWithConfig:
    """Test that config-loaded tools execute correctly using Echo LLM."""

    @pytest.mark.asyncio
    async def test_tool_execution_direct(self, echo_config):
        """Test executing a tool loaded via direct instantiation."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "tools": [
                {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {"prefix": "exec", "multiplier": 2},
                }
            ],
        }

        bot = await DynaBot.from_config(config)

        # Get and execute the tool
        tools = list(bot.tool_registry)
        tool = tools[0]
        result = await tool.execute(value="test")

        # Verify execution with configured parameters
        assert result == "exec:testexec:test"  # multiplier=2

    @pytest.mark.asyncio
    async def test_tool_execution_xref(self, echo_config):
        """Test executing a tool loaded via xref."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "tool_definitions": {
                "my_tool": {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {"prefix": "xref", "multiplier": 3},
                }
            },
            "tools": ["xref:tools[my_tool]"],
        }

        bot = await DynaBot.from_config(config)

        # Get and execute the tool
        tools = list(bot.tool_registry)
        tool = tools[0]
        result = await tool.execute(value="demo")

        # Verify execution with xref-configured parameters
        assert result == "xref:demoxref:demoxref:demo"  # multiplier=3


class TestConfigToolsWithReAct:
    """Test config-loaded tools with ReAct reasoning using Echo LLM."""

    @pytest.mark.asyncio
    async def test_react_with_config_tools(self, echo_config):
        """Test ReAct agent with config-loaded tools."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 3,
                "verbose": False,
                "store_trace": True,
            },
            "tools": [
                {
                    "class": "tests.fixtures.test_tools.SimpleTestTool",
                    "params": {},
                }
            ],
        }

        bot = await DynaBot.from_config(config)

        context = BotContext(
            conversation_id="test-react-config-tools",
            client_id="test-client",
        )

        # Generate response (may or may not use tool)
        response = await bot.chat("Hello!", context)

        # Verify response was generated
        assert response is not None
        assert isinstance(response, str)

        # Verify ReAct reasoning is enabled
        assert bot.reasoning_strategy is not None
        assert bot.reasoning_strategy.store_trace is True

    @pytest.mark.asyncio
    async def test_react_with_xref_tools(self, echo_config):
        """Test ReAct agent with xref-loaded tools."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "reasoning": {
                "strategy": "react",
                "max_iterations": 3,
                "verbose": False,
                "store_trace": True,
            },
            "tool_definitions": {
                "agent_tool": {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {"prefix": "agent"},
                }
            },
            "tools": ["xref:tools[agent_tool]"],
        }

        bot = await DynaBot.from_config(config)

        context = BotContext(
            conversation_id="test-react-xref-tools",
            client_id="test-client",
        )

        # Generate response
        response = await bot.chat("Hello!", context)

        # Verify response and tool availability
        assert response is not None

        # Verify tool was loaded correctly
        tools = list(bot.tool_registry)
        assert len(tools) == 1
        assert tools[0].prefix == "agent"


class TestErrorHandling:
    """Test error handling in config-based tool loading using Echo LLM."""

    @pytest.mark.asyncio
    async def test_invalid_tool_skipped(self, echo_config):
        """Test that invalid tools are skipped gracefully."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "tools": [
                {
                    "class": "tests.fixtures.test_tools.SimpleTestTool",
                    "params": {},
                },
                {
                    "class": "non.existent.Tool",  # Invalid
                    "params": {},
                },
            ],
        }

        bot = await DynaBot.from_config(config)

        # Should have 1 tool (invalid one skipped)
        tools = list(bot.tool_registry)
        assert len(tools) == 1
        assert isinstance(tools[0], SimpleTestTool)

    @pytest.mark.asyncio
    async def test_invalid_xref_skipped(self, echo_config):
        """Test that invalid xrefs are skipped gracefully."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "tool_definitions": {
                "valid": {
                    "class": "tests.fixtures.test_tools.SimpleTestTool",
                    "params": {},
                }
            },
            "tools": [
                "xref:tools[valid]",
                "xref:tools[nonexistent]",  # Invalid xref
            ],
        }

        bot = await DynaBot.from_config(config)

        # Should have 1 tool (invalid xref skipped)
        tools = list(bot.tool_registry)
        assert len(tools) == 1
        assert isinstance(tools[0], SimpleTestTool)

    @pytest.mark.asyncio
    async def test_bot_creation_succeeds_with_no_valid_tools(self, echo_config):
        """Test that bot creation succeeds even if no tools are valid."""
        config = {
            "llm": echo_config,
            "conversation_storage": {"backend": "memory"},
            "tools": [
                {"class": "non.existent.Tool1", "params": {}},
                {"class": "non.existent.Tool2", "params": {}},
            ],
        }

        # Should not raise error
        bot = await DynaBot.from_config(config)

        # Should have no tools
        tools = list(bot.tool_registry)
        assert len(tools) == 0
