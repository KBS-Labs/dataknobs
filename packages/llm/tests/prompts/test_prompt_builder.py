"""Unit tests for PromptBuilder."""

import pytest
from dataknobs_llm.prompts import (
    PromptBuilder,
    ConfigPromptLibrary,
    DictResourceAdapter,
    ValidationLevel,
)


class TestPromptBuilderInitialization:
    """Test suite for PromptBuilder initialization."""

    def test_initialization_basic(self):
        """Test basic initialization with library only."""
        library = ConfigPromptLibrary()
        builder = PromptBuilder(library=library)

        assert builder.library is library
        assert builder.adapters == {}
        assert builder._renderer is not None

    def test_initialization_with_adapters(self):
        """Test initialization with adapters."""
        library = ConfigPromptLibrary()
        adapter = DictResourceAdapter({"key": "value"})

        builder = PromptBuilder(
            library=library,
            adapters={"data": adapter}
        )

        assert "data" in builder.adapters
        assert builder.adapters["data"] is adapter

    def test_initialization_rejects_async_adapters(self):
        """Test that initialization rejects async adapters."""
        from dataknobs_llm.prompts import AsyncDictResourceAdapter

        library = ConfigPromptLibrary()
        async_adapter = AsyncDictResourceAdapter({"key": "value"})

        with pytest.raises(TypeError, match="async.*AsyncPromptBuilder"):
            PromptBuilder(
                library=library,
                adapters={"data": async_adapter}
            )

    def test_initialization_with_validation_level(self):
        """Test initialization with custom validation level."""
        library = ConfigPromptLibrary()
        builder = PromptBuilder(
            library=library,
            default_validation=ValidationLevel.ERROR
        )

        assert builder._renderer._default_validation == ValidationLevel.ERROR


class TestPromptBuilderSystemPrompts:
    """Test suite for rendering system prompts."""

    def test_render_system_prompt_basic(self):
        """Test rendering a basic system prompt."""
        config = {
            "system": {
                "greet": {
                    "template": "Hello {{name}}!"
                }
            }
        }
        library = ConfigPromptLibrary(config)
        builder = PromptBuilder(library=library)

        result = builder.render_system_prompt(
            "greet",
            params={"name": "Alice"}
        )

        assert result.content == "Hello Alice!"
        assert "name" in result.params_used
        assert result.params_used["name"] == "Alice"

    def test_render_system_prompt_with_defaults(self):
        """Test rendering system prompt with default values."""
        config = {
            "system": {
                "greet": {
                    "template": "Hello {{name}}!",
                    "defaults": {"name": "World"}
                }
            }
        }
        library = ConfigPromptLibrary(config)
        builder = PromptBuilder(library=library)

        # Without params, should use default
        result = builder.render_system_prompt("greet")
        assert result.content == "Hello World!"

        # With params, should override default
        result = builder.render_system_prompt("greet", params={"name": "Alice"})
        assert result.content == "Hello Alice!"

    def test_render_system_prompt_with_conditionals(self):
        """Test rendering system prompt with conditional sections."""
        config = {
            "system": {
                "greet": {
                    "template": "Hello {{name}}((, you are {{age}} years old))!"
                }
            }
        }
        library = ConfigPromptLibrary(config)
        builder = PromptBuilder(library=library)

        # With all params
        result = builder.render_system_prompt(
            "greet",
            params={"name": "Alice", "age": 30}
        )
        assert "you are 30 years old" in result.content

        # Without age (conditional should be removed)
        result = builder.render_system_prompt(
            "greet",
            params={"name": "Alice"}
        )
        assert "you are" not in result.content
        assert result.content == "Hello Alice!"

    def test_render_system_prompt_not_found(self):
        """Test error when system prompt not found."""
        library = ConfigPromptLibrary()
        builder = PromptBuilder(library=library)

        with pytest.raises(ValueError, match="not found"):
            builder.render_system_prompt("nonexistent")

    def test_render_system_prompt_with_validation_error(self):
        """Test rendering with validation error."""
        config = {
            "system": {
                "analyze": {
                    "template": "Analyze {{code}}",
                    "validation": {
                        "level": "error",
                        "required_params": ["code"]
                    }
                }
            }
        }
        library = ConfigPromptLibrary(config)
        builder = PromptBuilder(library=library)

        # Should raise error for missing required param
        with pytest.raises(ValueError, match="Missing required parameters"):
            builder.render_system_prompt("analyze")

        # Should succeed with required param
        result = builder.render_system_prompt(
            "analyze",
            params={"code": "print('hello')"}
        )
        assert result.content == "Analyze print('hello')"

    def test_render_system_prompt_with_validation_override(self):
        """Test rendering with validation level override."""
        config = {
            "system": {
                "analyze": {
                    "template": "Analyze {{code}}",
                    "validation": {
                        "level": "error",
                        "required_params": ["code"]
                    }
                }
            }
        }
        library = ConfigPromptLibrary(config)
        builder = PromptBuilder(library=library)

        # Override ERROR to IGNORE
        result = builder.render_system_prompt(
            "analyze",
            validation_override=ValidationLevel.IGNORE
        )

        # Should not raise error (missing var stays as-is in template)
        assert result.content == "Analyze {{code}}"


class TestPromptBuilderUserPrompts:
    """Test suite for rendering user prompts."""

    def test_render_user_prompt_basic(self):
        """Test rendering a basic user prompt."""
        config = {
            "user": {
                "ask": {
                    "template": "Tell me about {{topic}}"
                }
            }
        }
        library = ConfigPromptLibrary(config)
        builder = PromptBuilder(library=library)

        result = builder.render_user_prompt(
            "ask",
            params={"topic": "Python"}
        )

        assert result.content == "Tell me about Python"

    def test_render_user_prompt_multiple_variants(self):
        """Test rendering different user prompt variants."""
        config = {
            "user": {
                "ask": {"template": "First: {{question}}"},
                "ask_followup": {"template": "Follow-up: {{question}}"}
            }
        }
        library = ConfigPromptLibrary(config)
        builder = PromptBuilder(library=library)

        result0 = builder.render_user_prompt("ask", params={"question": "What?"})
        assert result0.content == "First: What?"

        result1 = builder.render_user_prompt("ask_followup", params={"question": "Why?"})
        assert result1.content == "Follow-up: Why?"

    def test_render_user_prompt_not_found(self):
        """Test error when user prompt not found."""
        library = ConfigPromptLibrary()
        builder = PromptBuilder(library=library)

        with pytest.raises(ValueError, match="not found"):
            builder.render_user_prompt("nonexistent")


class TestPromptBuilderRAG:
    """Test suite for RAG integration."""

    def test_render_with_rag_simple(self):
        """Test rendering with simple RAG configuration."""
        config = {
            "system": {
                "analyze": {
                    "template": "Analyze {{code}}\n\n{{RAG_CONTENT}}",
                    "rag_config_refs": ["code_docs"]  # Add reference in template
                }
            },
            "rag": {
                "code_docs": {
                    "adapter_name": "docs",
                    "query": "{{language}} documentation",
                    "k": 3
                }
            }
        }

        # Create library with RAG reference
        library = ConfigPromptLibrary(config)

        # Create adapter with search results
        search_results = [
            {"content": "Doc 1", "score": 0.9, "metadata": {}},
            {"content": "Doc 2", "score": 0.8, "metadata": {}},
        ]

        class MockAdapter(DictResourceAdapter):
            def search(self, query, k=5, filters=None, **kwargs):
                return search_results

        builder = PromptBuilder(
            library=library,
            adapters={"docs": MockAdapter({})}
        )

        result = builder.render_system_prompt(
            "analyze",
            params={"code": "test", "language": "python"}
        )

        # Should contain both template content and RAG content
        assert "Analyze test" in result.content
        assert "Doc 1" in result.content
        assert "Doc 2" in result.content

    def test_render_with_rag_formatted(self):
        """Test rendering with formatted RAG results."""
        config = {
            "system": {
                "analyze": {
                    "template": "{{RAG_CONTENT}}",
                    "rag_config_refs": ["docs"]  # Add reference in template
                }
            },
            "rag": {
                "docs": {
                    "adapter_name": "docs",
                    "query": "test",
                    "k": 2,
                    "header": "# Documentation\n",
                    "item_template": "{{index}}. {{content}}\n"
                }
            }
        }

        library = ConfigPromptLibrary(config)

        search_results = [
            {"content": "First doc", "score": 0.9, "metadata": {}},
            {"content": "Second doc", "score": 0.8, "metadata": {}},
        ]

        class MockAdapter(DictResourceAdapter):
            def search(self, query, k=5, filters=None, **kwargs):
                return search_results

        builder = PromptBuilder(
            library=library,
            adapters={"docs": MockAdapter({})}
        )

        result = builder.render_system_prompt("analyze")

        # Check formatting
        assert "# Documentation" in result.content
        assert "1. First doc" in result.content
        assert "2. Second doc" in result.content

    def test_render_without_rag(self):
        """Test rendering with RAG disabled."""
        config = {
            "system": {
                "analyze": {
                    "template": "Analyze {{code}}\n\n{{RAG_CONTENT}}"
                }
            }
        }

        library = ConfigPromptLibrary(config)
        builder = PromptBuilder(library=library)

        result = builder.render_system_prompt(
            "analyze",
            params={"code": "test"},
            include_rag=False
        )

        # RAG_CONTENT should be empty/missing
        assert "Analyze test" in result.content

    def test_render_rag_adapter_not_found(self):
        """Test error handling when RAG adapter not found."""
        config = {
            "system": {
                "analyze": {
                    "template": "{{RAG_CONTENT}}",
                    "rag_config_refs": ["docs"]  # Add reference in template
                }
            },
            "rag": {
                "docs": {
                    "adapter_name": "nonexistent",
                    "query": "test"
                }
            }
        }

        library = ConfigPromptLibrary(config)

        # Set raise_on_rag_error=True to get exception instead of warning
        builder = PromptBuilder(library=library, raise_on_rag_error=True)

        with pytest.raises((KeyError, RuntimeError), match="Adapter.*not found|RAG search failed"):
            builder.render_system_prompt("analyze")

    def test_render_rag_error_handling(self):
        """Test RAG error handling with raise_on_rag_error=False."""
        config = {
            "system": {
                "analyze": {
                    "template": "{{RAG_CONTENT}}",
                    "rag_config_refs": ["docs"]  # Add reference in template
                }
            },
            "rag": {
                "docs": {
                    "adapter_name": "docs",
                    "query": "test"
                }
            }
        }

        library = ConfigPromptLibrary(config)

        class FailingAdapter(DictResourceAdapter):
            def search(self, query, k=5, filters=None, **kwargs):
                raise RuntimeError("Search failed")

        builder = PromptBuilder(
            library=library,
            adapters={"docs": FailingAdapter({})},
            raise_on_rag_error=False  # Should log warning, not raise
        )

        result = builder.render_system_prompt("analyze")

        # Should have empty RAG content but not fail
        assert result.content == ""

    def test_render_rag_error_raising(self):
        """Test RAG error handling with raise_on_rag_error=True."""
        config = {
            "system": {
                "analyze": {
                    "template": "{{RAG_CONTENT}}",
                    "rag_config_refs": ["docs"]  # Add reference in template
                }
            },
            "rag": {
                "docs": {
                    "adapter_name": "docs",
                    "query": "test"
                }
            }
        }

        library = ConfigPromptLibrary(config)

        class FailingAdapter(DictResourceAdapter):
            def search(self, query, k=5, filters=None, **kwargs):
                raise RuntimeError("Search failed")

        builder = PromptBuilder(
            library=library,
            adapters={"docs": FailingAdapter({})},
            raise_on_rag_error=True  # Should raise error
        )

        with pytest.raises(RuntimeError, match="RAG search failed"):
            builder.render_system_prompt("analyze")


class TestPromptBuilderHelpers:
    """Test suite for helper methods."""

    def test_get_required_parameters_system(self):
        """Test getting required parameters for system prompt."""
        config = {
            "system": {
                "analyze": {
                    "template": "Analyze {{code}}",
                    "validation": {
                        "level": "error",
                        "required_params": ["code", "language"]
                    }
                }
            }
        }
        library = ConfigPromptLibrary(config)
        builder = PromptBuilder(library=library)

        required = builder.get_required_parameters("analyze", prompt_type="system")

        assert "code" in required
        assert "language" in required

    def test_get_required_parameters_user(self):
        """Test getting required parameters for user prompt."""
        config = {
            "user": {
                "ask": {
                    "template": "{{question}}",
                    "validation": {
                        "required_params": ["question"]
                    }
                }
            }
        }
        library = ConfigPromptLibrary(config)
        builder = PromptBuilder(library=library)

        required = builder.get_required_parameters("ask", prompt_type="user")

        assert "question" in required

    def test_get_required_parameters_not_found(self):
        """Test error when prompt not found."""
        library = ConfigPromptLibrary()
        builder = PromptBuilder(library=library)

        with pytest.raises(ValueError, match="not found"):
            builder.get_required_parameters("nonexistent")

    def test_repr(self):
        """Test string representation."""
        library = ConfigPromptLibrary()
        adapter = DictResourceAdapter({"key": "value"})
        builder = PromptBuilder(
            library=library,
            adapters={"data": adapter}
        )

        repr_str = repr(builder)

        assert "PromptBuilder" in repr_str
        assert "data" in repr_str


class TestPromptBuilderIntegration:
    """Integration tests for PromptBuilder."""

    def test_end_to_end_with_defaults_and_rag(self):
        """Test complete workflow with defaults, params, and RAG."""
        config = {
            "system": {
                "analyze": {
                    "template": "Analyze {{language}} code: {{code}}\n\n{{RAG_CONTENT}}",
                    "defaults": {"language": "python"},
                    "validation": {
                        "level": "warn",
                        "required_params": ["code"]
                    },
                    "rag_config_refs": ["docs"]  # Add reference in template
                }
            },
            "rag": {
                "docs": {
                    "adapter_name": "docs",
                    "query": "{{language}} best practices",
                    "k": 2,
                    "header": "Relevant docs:\n",
                    "item_template": "- {{content}}\n"
                }
            }
        }

        library = ConfigPromptLibrary(config)

        search_results = [
            {"content": "Use type hints", "score": 0.9, "metadata": {}},
            {"content": "Follow PEP 8", "score": 0.8, "metadata": {}},
        ]

        class MockAdapter(DictResourceAdapter):
            def search(self, query, k=5, filters=None, **kwargs):
                assert "python" in query.lower()
                return search_results

        builder = PromptBuilder(
            library=library,
            adapters={"docs": MockAdapter({})}
        )

        result = builder.render_system_prompt(
            "analyze",
            params={"code": "def foo(): pass"}
        )

        # Check all components are present
        assert "Analyze python code" in result.content  # Used default
        assert "def foo(): pass" in result.content
        assert "Relevant docs:" in result.content
        assert "Use type hints" in result.content
        assert "Follow PEP 8" in result.content

        # Check metadata
        assert result.metadata["prompt_name"] == "analyze"
        assert result.metadata["prompt_type"] == "system"
        assert result.metadata["include_rag"] is True
