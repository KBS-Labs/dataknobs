"""Unit tests for tool resolution from configuration."""

import pytest
from dataknobs_common.exceptions import ConfigurationError

from dataknobs_bots import DynaBot
from tests.fixtures.test_tools import (
    KBDependentTestTool,
    ParameterizedTestTool,
    SimpleTestTool,
)


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
        """Invalid class path raises ConfigurationError."""
        tool_config = {
            "class": "non.existent.Module",
            "params": {},
        }

        with pytest.raises(ConfigurationError, match="Failed to import tool class"):
            DynaBot._resolve_tool(tool_config, {})

    def test_resolve_invalid_class_path_optional(self):
        """Invalid class path with optional: true returns None."""
        tool_config = {
            "class": "non.existent.Module",
            "params": {},
            "optional": True,
        }

        tool = DynaBot._resolve_tool(tool_config, {})
        assert tool is None

    def test_resolve_invalid_xref(self):
        """Non-existent xref target raises ConfigurationError."""
        config = {
            "tool_definitions": {
                "existing_tool": {
                    "class": "tests.fixtures.test_tools.SimpleTestTool",
                }
            }
        }

        tool_config = "xref:tools[non_existent]"

        with pytest.raises(ConfigurationError, match="Tool definition not found"):
            DynaBot._resolve_tool(tool_config, config)

    def test_resolve_invalid_xref_format(self):
        """Malformed xref string raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Invalid xref format"):
            DynaBot._resolve_tool("xref:invalid_format", {})

    def test_resolve_non_tool_class(self):
        """Class that isn't a Tool raises ConfigurationError."""
        tool_config = {
            "class": "builtins.str",  # Not a Tool subclass
            "params": {},
        }

        with pytest.raises(ConfigurationError, match="not a Tool instance"):
            DynaBot._resolve_tool(tool_config, {})

    def test_resolve_missing_class_key(self):
        """Config without class or xref key raises ConfigurationError."""
        tool_config = {
            "name": "something",
            "other": "data",
        }

        with pytest.raises(ConfigurationError, match="Invalid tool config format"):
            DynaBot._resolve_tool(tool_config, {})

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
    async def test_from_config_skips_optional_invalid_tools(self):
        """Bot creation continues when optional tools fail to resolve."""
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
                    "class": "non.existent.Tool",
                    "params": {},
                    "optional": True,
                },
                {
                    "class": "tests.fixtures.test_tools.ParameterizedTestTool",
                    "params": {},
                },
            ],
        }

        bot = await DynaBot.from_config(config)

        # Should have 2 tools (skipped the optional invalid one)
        tools = list(bot.tool_registry)
        assert len(tools) == 2

        # Verify correct tools were loaded
        tool_types = {type(tool) for tool in tools}
        assert SimpleTestTool in tool_types
        assert ParameterizedTestTool in tool_types

    @pytest.mark.asyncio
    async def test_from_config_raises_on_required_invalid_tool(self):
        """Bot creation fails when a required tool cannot be resolved."""
        config = {
            "llm": {
                "provider": "echo",
                "model": "test",
            },
            "conversation_storage": {
                "backend": "memory",
            },
            "tools": [
                {
                    "class": "non.existent.Tool",
                    "params": {},
                },
            ],
        }

        with pytest.raises(ConfigurationError, match="Failed to import tool class"):
            await DynaBot.from_config(config)

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

    def test_resolve_tool_with_kb_dependency_injection(self):
        """Bug: tools declaring requires=('knowledge_base',) fail to instantiate.

        _resolve_tool does not inject dependencies declared in
        catalog_metadata().requires, so KBDependentTestTool (and
        KnowledgeSearchTool) raises TypeError because knowledge_base
        is a required constructor argument.
        """
        sentinel_kb = object()  # stand-in for a real knowledge base
        tool_config = {
            "class": "tests.fixtures.test_tools.KBDependentTestTool",
            "params": {},
        }
        config = {}
        dependencies = {"knowledge_base": sentinel_kb}

        tool = DynaBot._resolve_tool(tool_config, config, dependencies=dependencies)

        assert tool is not None
        assert isinstance(tool, KBDependentTestTool)
        assert tool.knowledge_base is sentinel_kb

    def test_resolve_tool_without_dependency_still_works(self):
        """Tools that do NOT declare requires should be unaffected."""
        tool_config = {
            "class": "tests.fixtures.test_tools.SimpleTestTool",
            "params": {},
        }
        config = {}
        dependencies = {"knowledge_base": object()}

        tool = DynaBot._resolve_tool(tool_config, config, dependencies=dependencies)

        assert tool is not None
        assert isinstance(tool, SimpleTestTool)

    @pytest.mark.asyncio
    async def test_from_config_injects_kb_into_tool(self):
        """Bug: from_config creates tools before KB, so KB-dependent tools fail.

        After fix, from_config should:
        1. Create KB before tools
        2. Inject KB into tools that declare the dependency
        """
        config = {
            "llm": {
                "provider": "echo",
                "model": "test",
            },
            "conversation_storage": {
                "backend": "memory",
            },
            "knowledge_base": {
                "enabled": True,
                "type": "rag",
                "vector_store": {
                    "backend": "memory",
                    "dimensions": 384,
                },
                "embedding_provider": "echo",
                "embedding_model": "test",
            },
            "tools": [
                {
                    "class": "tests.fixtures.test_tools.KBDependentTestTool",
                    "params": {},
                },
            ],
        }

        bot = await DynaBot.from_config(config)

        # Tool should have been created and registered
        tools = list(bot.tool_registry)
        assert len(tools) == 1

        tool = tools[0]
        assert isinstance(tool, KBDependentTestTool)
        # Tool should have received the knowledge base
        assert tool.knowledge_base is not None
        assert tool.knowledge_base is bot.knowledge_base
