"""Unit tests for CompositePromptLibrary."""

import pytest
from dataknobs_llm.prompts import (
    CompositePromptLibrary,
    ConfigPromptLibrary,
)


class TestCompositeLibraryInit:
    """Test suite for CompositePromptLibrary initialization."""

    def test_initialization_empty(self):
        """Test initialization with no libraries."""
        library = CompositePromptLibrary()
        assert len(library.libraries) == 0
        assert library.get_system_prompt("any") is None

    def test_initialization_with_libraries(self):
        """Test initialization with library list."""
        lib1 = ConfigPromptLibrary({"system": {"test": {"template": "Test"}}})
        lib2 = ConfigPromptLibrary({})

        composite = CompositePromptLibrary(libraries=[lib1, lib2])
        assert len(composite.libraries) == 2

    def test_initialization_with_names(self):
        """Test initialization with library names."""
        lib1 = ConfigPromptLibrary({})
        lib2 = ConfigPromptLibrary({})

        composite = CompositePromptLibrary(
            libraries=[lib1, lib2],
            names=["first", "second"]
        )

        assert composite.library_names == ["first", "second"]

    def test_initialization_mismatched_names(self):
        """Test initialization with mismatched library/name counts."""
        lib1 = ConfigPromptLibrary({})
        lib2 = ConfigPromptLibrary({})

        with pytest.raises(ValueError, match="Number of names"):
            CompositePromptLibrary(
                libraries=[lib1, lib2],
                names=["only_one"]
            )

    def test_default_library_names(self):
        """Test that default names are generated."""
        lib1 = ConfigPromptLibrary({})
        lib2 = ConfigPromptLibrary({})

        composite = CompositePromptLibrary(libraries=[lib1, lib2])
        assert composite.library_names == ["library_0", "library_1"]


class TestCompositeSystemPrompts:
    """Test suite for system prompts in CompositePromptLibrary."""

    def test_system_prompt_from_first_library(self):
        """Test getting system prompt from first library."""
        lib1 = ConfigPromptLibrary({
            "system": {"test": {"template": "From lib1"}}
        })
        lib2 = ConfigPromptLibrary({
            "system": {"test": {"template": "From lib2"}}
        })

        composite = CompositePromptLibrary(libraries=[lib1, lib2])
        template = composite.get_system_prompt("test")

        # Should get from first library
        assert template["template"] == "From lib1"

    def test_system_prompt_fallback_to_second_library(self):
        """Test fallback to second library when not in first."""
        lib1 = ConfigPromptLibrary({"system": {}})
        lib2 = ConfigPromptLibrary({
            "system": {"test": {"template": "From lib2"}}
        })

        composite = CompositePromptLibrary(libraries=[lib1, lib2])
        template = composite.get_system_prompt("test")

        # Should fall back to second library
        assert template["template"] == "From lib2"

    def test_system_prompt_not_in_any_library(self):
        """Test system prompt not found in any library."""
        lib1 = ConfigPromptLibrary({"system": {}})
        lib2 = ConfigPromptLibrary({"system": {}})

        composite = CompositePromptLibrary(libraries=[lib1, lib2])
        template = composite.get_system_prompt("nonexistent")

        assert template is None

    def test_system_prompt_priority_order(self):
        """Test that libraries are searched in priority order."""
        lib1 = ConfigPromptLibrary({
            "system": {"a": {"template": "A from lib1"}}
        })
        lib2 = ConfigPromptLibrary({
            "system": {
                "a": {"template": "A from lib2"},
                "b": {"template": "B from lib2"}
            }
        })
        lib3 = ConfigPromptLibrary({
            "system": {
                "a": {"template": "A from lib3"},
                "b": {"template": "B from lib3"},
                "c": {"template": "C from lib3"}
            }
        })

        composite = CompositePromptLibrary(libraries=[lib1, lib2, lib3])

        # 'a' should come from lib1 (highest priority)
        assert composite.get_system_prompt("a")["template"] == "A from lib1"

        # 'b' should come from lib2 (lib1 doesn't have it)
        assert composite.get_system_prompt("b")["template"] == "B from lib2"

        # 'c' should come from lib3 (only lib3 has it)
        assert composite.get_system_prompt("c")["template"] == "C from lib3"


class TestCompositeUserPrompts:
    """Test suite for user prompts in CompositePromptLibrary."""

    def test_user_prompt_fallback(self):
        """Test user prompt with fallback behavior."""
        lib1 = ConfigPromptLibrary({
            "user": {"question": {"template": "From lib1"}}
        })
        lib2 = ConfigPromptLibrary({
            "user": {"question": {"template": "From lib2"}}
        })

        composite = CompositePromptLibrary(libraries=[lib1, lib2])
        template = composite.get_user_prompt("question")

        assert template["template"] == "From lib1"

    def test_user_prompt_different_names_different_libraries(self):
        """Test different user prompts from different libraries."""
        lib1 = ConfigPromptLibrary({
            "user": {"question": {"template": "Question from lib1"}}
        })
        lib2 = ConfigPromptLibrary({
            "user": {"question_alt": {"template": "Alt question from lib2"}}
        })

        composite = CompositePromptLibrary(libraries=[lib1, lib2])

        template0 = composite.get_user_prompt("question")
        template1 = composite.get_user_prompt("question_alt")

        assert template0["template"] == "Question from lib1"
        assert template1["template"] == "Alt question from lib2"


class TestCompositeMessageIndexes:
    """Test suite for message indexes in CompositePromptLibrary."""

    def test_message_index_fallback(self):
        """Test message index with fallback behavior."""
        lib1 = ConfigPromptLibrary({"messages": {}})
        lib2 = ConfigPromptLibrary({
            "messages": {
                "conversation": {
                    "messages": [{"role": "user", "content": "Hello"}]
                }
            }
        })

        composite = CompositePromptLibrary(libraries=[lib1, lib2])
        index = composite.get_message_index("conversation")

        assert index is not None
        assert len(index["messages"]) == 1


class TestCompositeRAGConfigs:
    """Test suite for RAG configs in CompositePromptLibrary."""

    def test_rag_config_fallback(self):
        """Test RAG config with fallback behavior."""
        lib1 = ConfigPromptLibrary({"rag": {}})
        lib2 = ConfigPromptLibrary({
            "rag": {
                "docs_search": {
                    "adapter_name": "docs",
                    "query": "{{q}}"
                }
            }
        })

        composite = CompositePromptLibrary(libraries=[lib1, lib2])
        rag = composite.get_rag_config("docs_search")

        assert rag is not None
        assert rag["adapter_name"] == "docs"


class TestCompositeLibraryManagement:
    """Test suite for library management operations."""

    def test_add_library_to_end(self):
        """Test adding library to end of list."""
        lib1 = ConfigPromptLibrary({"system": {"a": {"template": "A"}}})
        composite = CompositePromptLibrary(libraries=[lib1])

        lib2 = ConfigPromptLibrary({"system": {"b": {"template": "B"}}})
        composite.add_library(lib2, name="second")

        assert len(composite.libraries) == 2
        assert composite.library_names[-1] == "second"

    def test_add_library_with_priority(self):
        """Test adding library at specific priority."""
        lib1 = ConfigPromptLibrary({"system": {"test": {"template": "Lib1"}}})
        lib2 = ConfigPromptLibrary({"system": {"test": {"template": "Lib2"}}})

        composite = CompositePromptLibrary(libraries=[lib1], names=["first"])

        # Add lib2 at priority 0 (highest priority, before lib1)
        composite.add_library(lib2, name="second", priority=0)

        assert composite.library_names[0] == "second"
        assert composite.library_names[1] == "first"

        # lib2 should be searched first
        template = composite.get_system_prompt("test")
        assert template["template"] == "Lib2"

    def test_add_library_default_name(self):
        """Test adding library without specifying name."""
        lib1 = ConfigPromptLibrary({})
        composite = CompositePromptLibrary()

        composite.add_library(lib1)
        assert "library_0" in composite.library_names

    def test_remove_library_by_name(self):
        """Test removing library by name."""
        lib1 = ConfigPromptLibrary({"system": {"a": {"template": "A"}}})
        lib2 = ConfigPromptLibrary({"system": {"b": {"template": "B"}}})

        composite = CompositePromptLibrary(
            libraries=[lib1, lib2],
            names=["first", "second"]
        )

        # Remove first library
        result = composite.remove_library("first")
        assert result is True
        assert len(composite.libraries) == 1
        assert "first" not in composite.library_names

    def test_remove_nonexistent_library(self):
        """Test removing library that doesn't exist."""
        lib1 = ConfigPromptLibrary({})
        composite = CompositePromptLibrary(libraries=[lib1], names=["first"])

        result = composite.remove_library("nonexistent")
        assert result is False
        assert len(composite.libraries) == 1

    def test_get_library_by_name(self):
        """Test getting specific library by name."""
        lib1 = ConfigPromptLibrary({"system": {"a": {"template": "A"}}})
        lib2 = ConfigPromptLibrary({"system": {"b": {"template": "B"}}})

        composite = CompositePromptLibrary(
            libraries=[lib1, lib2],
            names=["first", "second"]
        )

        retrieved = composite.get_library_by_name("second")
        assert retrieved is lib2

    def test_get_nonexistent_library_by_name(self):
        """Test getting library that doesn't exist."""
        composite = CompositePromptLibrary()
        assert composite.get_library_by_name("nonexistent") is None


class TestCompositeLibraryEdgeCases:
    """Test edge cases for CompositePromptLibrary."""

    def test_empty_composite_library(self):
        """Test composite library with no libraries."""
        composite = CompositePromptLibrary()

        assert composite.get_system_prompt("any") is None
        assert composite.get_user_prompt("any") is None
        assert composite.get_message_index("any") is None
        assert composite.get_rag_config("any") is None

    def test_override_pattern(self):
        """Test common override pattern (custom + defaults)."""
        # Default library with many prompts
        defaults = ConfigPromptLibrary({
            "system": {
                "greet": {"template": "Default greeting"},
                "analyze": {"template": "Default analysis"},
                "summarize": {"template": "Default summary"}
            }
        })

        # Custom library overriding specific prompts
        custom = ConfigPromptLibrary({
            "system": {
                "greet": {"template": "Custom greeting"}
            }
        })

        # Composite searches custom first, then defaults
        composite = CompositePromptLibrary(
            libraries=[custom, defaults],
            names=["custom", "defaults"]
        )

        # Should get custom greeting
        greet = composite.get_system_prompt("greet")
        assert greet["template"] == "Custom greeting"

        # Should get default analysis (not overridden)
        analyze = composite.get_system_prompt("analyze")
        assert analyze["template"] == "Default analysis"

    def test_library_properties_are_copies(self):
        """Test that library property returns copies, not originals."""
        lib1 = ConfigPromptLibrary({})
        composite = CompositePromptLibrary(libraries=[lib1])

        # Get libraries property
        libraries = composite.libraries

        # Modify the returned list
        libraries.append(ConfigPromptLibrary({}))

        # Should not affect the composite
        assert len(composite.libraries) == 1

    def test_multiple_fallback_layers(self):
        """Test with many fallback layers."""
        lib1 = ConfigPromptLibrary({"system": {"a": {"template": "A1"}}})
        lib2 = ConfigPromptLibrary({"system": {"b": {"template": "B2"}}})
        lib3 = ConfigPromptLibrary({"system": {"c": {"template": "C3"}}})
        lib4 = ConfigPromptLibrary({"system": {"d": {"template": "D4"}}})

        composite = CompositePromptLibrary(libraries=[lib1, lib2, lib3, lib4])

        # Each prompt should come from its respective library
        assert composite.get_system_prompt("a")["template"] == "A1"
        assert composite.get_system_prompt("b")["template"] == "B2"
        assert composite.get_system_prompt("c")["template"] == "C3"
        assert composite.get_system_prompt("d")["template"] == "D4"
