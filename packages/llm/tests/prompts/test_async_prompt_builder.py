"""Unit tests for AsyncPromptBuilder."""

import pytest
import asyncio
from dataknobs_llm.prompts import (
    AsyncPromptBuilder,
    ConfigPromptLibrary,
    AsyncDictResourceAdapter,
    ValidationLevel,
)


class TestAsyncPromptBuilderInitialization:
    """Test suite for AsyncPromptBuilder initialization."""

    def test_initialization_basic(self):
        """Test basic initialization with library only."""
        library = ConfigPromptLibrary()
        builder = AsyncPromptBuilder(library=library)

        assert builder.library is library
        assert builder.adapters == {}
        assert builder._renderer is not None

    def test_initialization_with_adapters(self):
        """Test initialization with adapters."""
        library = ConfigPromptLibrary()
        adapter = AsyncDictResourceAdapter({"key": "value"})

        builder = AsyncPromptBuilder(
            library=library,
            adapters={"data": adapter}
        )

        assert "data" in builder.adapters
        assert builder.adapters["data"] is adapter

    def test_initialization_rejects_sync_adapters(self):
        """Test that initialization rejects sync adapters."""
        from dataknobs_llm.prompts import DictResourceAdapter

        library = ConfigPromptLibrary()
        sync_adapter = DictResourceAdapter({"key": "value"})

        with pytest.raises(TypeError, match="synchronous.*PromptBuilder"):
            AsyncPromptBuilder(
                library=library,
                adapters={"data": sync_adapter}
            )

    def test_initialization_with_validation_level(self):
        """Test initialization with custom validation level."""
        library = ConfigPromptLibrary()
        builder = AsyncPromptBuilder(
            library=library,
            default_validation=ValidationLevel.ERROR
        )

        assert builder._renderer._default_validation == ValidationLevel.ERROR


@pytest.mark.asyncio
class TestAsyncPromptBuilderSystemPrompts:
    """Test suite for rendering system prompts asynchronously."""

    async def test_render_system_prompt_basic(self):
        """Test rendering a basic system prompt."""
        config = {
            "system": {
                "greet": {
                    "template": "Hello {{name}}!"
                }
            }
        }
        library = ConfigPromptLibrary(config)
        builder = AsyncPromptBuilder(library=library)

        result = await builder.render_system_prompt(
            "greet",
            params={"name": "Alice"}
        )

        assert result.content == "Hello Alice!"
        assert "name" in result.params_used
        assert result.params_used["name"] == "Alice"

    async def test_render_system_prompt_with_defaults(self):
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
        builder = AsyncPromptBuilder(library=library)

        # Without params, should use default
        result = await builder.render_system_prompt("greet")
        assert result.content == "Hello World!"

        # With params, should override default
        result = await builder.render_system_prompt("greet", params={"name": "Alice"})
        assert result.content == "Hello Alice!"

    async def test_render_system_prompt_with_conditionals(self):
        """Test rendering system prompt with conditional sections."""
        config = {
            "system": {
                "greet": {
                    "template": "Hello {{name}}((, you are {{age}} years old))!"
                }
            }
        }
        library = ConfigPromptLibrary(config)
        builder = AsyncPromptBuilder(library=library)

        # With all params
        result = await builder.render_system_prompt(
            "greet",
            params={"name": "Alice", "age": 30}
        )
        assert "you are 30 years old" in result.content

        # Without age (conditional should be removed)
        result = await builder.render_system_prompt(
            "greet",
            params={"name": "Alice"}
        )
        assert "you are" not in result.content
        assert result.content == "Hello Alice!"

    async def test_render_system_prompt_not_found(self):
        """Test error when system prompt not found."""
        library = ConfigPromptLibrary()
        builder = AsyncPromptBuilder(library=library)

        with pytest.raises(ValueError, match="not found"):
            await builder.render_system_prompt("nonexistent")

    async def test_render_system_prompt_with_validation_error(self):
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
        builder = AsyncPromptBuilder(library=library)

        # Should raise error for missing required param
        with pytest.raises(ValueError, match="Missing required parameters"):
            await builder.render_system_prompt("analyze")

        # Should succeed with required param
        result = await builder.render_system_prompt(
            "analyze",
            params={"code": "print('hello')"}
        )
        assert result.content == "Analyze print('hello')"

    async def test_render_system_prompt_with_validation_override(self):
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
        builder = AsyncPromptBuilder(library=library)

        # Override ERROR to IGNORE
        result = await builder.render_system_prompt(
            "analyze",
            validation_override=ValidationLevel.IGNORE
        )

        # Should not raise error (missing var stays as-is in template)
        assert result.content == "Analyze {{code}}"


@pytest.mark.asyncio
class TestAsyncPromptBuilderUserPrompts:
    """Test suite for rendering user prompts asynchronously."""

    async def test_render_user_prompt_basic(self):
        """Test rendering a basic user prompt."""
        config = {
            "user": {
                "ask": {
                    0: {
                        "template": "Tell me about {{topic}}"
                    }
                }
            }
        }
        library = ConfigPromptLibrary(config)
        builder = AsyncPromptBuilder(library=library)

        result = await builder.render_user_prompt(
            "ask",
            index=0,
            params={"topic": "Python"}
        )

        assert result.content == "Tell me about Python"
        assert result.metadata["index"] == 0

    async def test_render_user_prompt_multiple_variants(self):
        """Test rendering different user prompt variants."""
        config = {
            "user": {
                "ask": {
                    0: {"template": "First: {{question}}"},
                    1: {"template": "Follow-up: {{question}}"}
                }
            }
        }
        library = ConfigPromptLibrary(config)
        builder = AsyncPromptBuilder(library=library)

        result0 = await builder.render_user_prompt("ask", index=0, params={"question": "What?"})
        assert result0.content == "First: What?"

        result1 = await builder.render_user_prompt("ask", index=1, params={"question": "Why?"})
        assert result1.content == "Follow-up: Why?"

    async def test_render_user_prompt_not_found(self):
        """Test error when user prompt not found."""
        library = ConfigPromptLibrary()
        builder = AsyncPromptBuilder(library=library)

        with pytest.raises(ValueError, match="not found"):
            await builder.render_user_prompt("nonexistent")


@pytest.mark.asyncio
class TestAsyncPromptBuilderRAG:
    """Test suite for RAG integration with async."""

    async def test_render_with_rag_simple(self):
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

        library = ConfigPromptLibrary(config)

        search_results = [
            {"content": "Doc 1", "score": 0.9, "metadata": {}},
            {"content": "Doc 2", "score": 0.8, "metadata": {}},
        ]

        class MockAsyncAdapter(AsyncDictResourceAdapter):
            async def search(self, query, k=5, filters=None, **kwargs):
                await asyncio.sleep(0.01)  # Simulate async operation
                return search_results

        builder = AsyncPromptBuilder(
            library=library,
            adapters={"docs": MockAsyncAdapter({})}
        )

        result = await builder.render_system_prompt(
            "analyze",
            params={"code": "test", "language": "python"}
        )

        # Should contain both template content and RAG content
        assert "Analyze test" in result.content
        assert "Doc 1" in result.content
        assert "Doc 2" in result.content

    async def test_render_with_rag_formatted(self):
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

        class MockAsyncAdapter(AsyncDictResourceAdapter):
            async def search(self, query, k=5, filters=None, **kwargs):
                return search_results

        builder = AsyncPromptBuilder(
            library=library,
            adapters={"docs": MockAsyncAdapter({})}
        )

        result = await builder.render_system_prompt("analyze")

        # Check formatting
        assert "# Documentation" in result.content
        assert "1. First doc" in result.content
        assert "2. Second doc" in result.content

    async def test_render_with_parallel_rag(self):
        """Test that multiple RAG searches execute in parallel."""
        config = {
            "system": {
                "analyze": {
                    "template": "{{RAG_1}}\n{{RAG_2}}",
                    "rag_config_refs": ["docs1", "docs2"]  # Add reference in template
                }
            },
            "rag": {
                "docs1": {
                    "adapter_name": "docs1",
                    "query": "test1",
                    "placeholder": "RAG_1"
                },
                "docs2": {
                    "adapter_name": "docs2",
                    "query": "test2",
                    "placeholder": "RAG_2"
                }
            }
        }

        library = ConfigPromptLibrary(config)

        search_count = {"count": 0}

        class MockAsyncAdapter(AsyncDictResourceAdapter):
            def __init__(self, data, result_text):
                super().__init__(data)
                self.result_text = result_text

            async def search(self, query, k=5, filters=None, **kwargs):
                search_count["count"] += 1
                await asyncio.sleep(0.05)  # Simulate slow search
                return [{"content": self.result_text, "score": 1.0, "metadata": {}}]

        import time
        start_time = time.time()

        builder = AsyncPromptBuilder(
            library=library,
            adapters={
                "docs1": MockAsyncAdapter({}, "Result 1"),
                "docs2": MockAsyncAdapter({}, "Result 2")
            }
        )

        result = await builder.render_system_prompt("analyze")

        elapsed = time.time() - start_time

        # Both searches should have executed
        assert search_count["count"] == 2
        assert "Result 1" in result.content
        assert "Result 2" in result.content

        # Should take ~0.05s (parallel), not ~0.1s (sequential)
        assert elapsed < 0.08, f"Took {elapsed}s, expected < 0.08s (parallel execution)"

    async def test_render_without_rag(self):
        """Test rendering with RAG disabled."""
        config = {
            "system": {
                "analyze": {
                    "template": "Analyze {{code}}\n\n{{RAG_CONTENT}}"
                }
            }
        }

        library = ConfigPromptLibrary(config)
        builder = AsyncPromptBuilder(library=library)

        result = await builder.render_system_prompt(
            "analyze",
            params={"code": "test"},
            include_rag=False
        )

        # RAG_CONTENT should be empty/missing
        assert "Analyze test" in result.content

    async def test_render_rag_adapter_not_found(self):
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
        builder = AsyncPromptBuilder(library=library, raise_on_rag_error=True)

        with pytest.raises((KeyError, RuntimeError), match="Adapter.*not found|RAG search failed"):
            await builder.render_system_prompt("analyze")

    async def test_render_rag_error_handling(self):
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

        class FailingAdapter(AsyncDictResourceAdapter):
            async def search(self, query, k=5, filters=None, **kwargs):
                raise RuntimeError("Search failed")

        builder = AsyncPromptBuilder(
            library=library,
            adapters={"docs": FailingAdapter({})},
            raise_on_rag_error=False  # Should log warning, not raise
        )

        result = await builder.render_system_prompt("analyze")

        # Should have empty RAG content but not fail
        assert result.content == ""

    async def test_render_rag_error_raising(self):
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

        class FailingAdapter(AsyncDictResourceAdapter):
            async def search(self, query, k=5, filters=None, **kwargs):
                raise RuntimeError("Search failed")

        builder = AsyncPromptBuilder(
            library=library,
            adapters={"docs": FailingAdapter({})},
            raise_on_rag_error=True  # Should raise error
        )

        with pytest.raises(RuntimeError, match="RAG search failed"):
            await builder.render_system_prompt("analyze")


@pytest.mark.asyncio
class TestAsyncPromptBuilderHelpers:
    """Test suite for helper methods."""

    async def test_get_required_parameters_system(self):
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
        builder = AsyncPromptBuilder(library=library)

        required = builder.get_required_parameters("analyze", prompt_type="system")

        assert "code" in required
        assert "language" in required

    async def test_get_required_parameters_user(self):
        """Test getting required parameters for user prompt."""
        config = {
            "user": {
                "ask": {
                    0: {
                        "template": "{{question}}",
                        "validation": {
                            "required_params": ["question"]
                        }
                    }
                }
            }
        }
        library = ConfigPromptLibrary(config)
        builder = AsyncPromptBuilder(library=library)

        required = builder.get_required_parameters("ask", prompt_type="user", index=0)

        assert "question" in required

    async def test_get_required_parameters_not_found(self):
        """Test error when prompt not found."""
        library = ConfigPromptLibrary()
        builder = AsyncPromptBuilder(library=library)

        with pytest.raises(ValueError, match="not found"):
            builder.get_required_parameters("nonexistent")

    async def test_repr(self):
        """Test string representation."""
        library = ConfigPromptLibrary()
        adapter = AsyncDictResourceAdapter({"key": "value"})
        builder = AsyncPromptBuilder(
            library=library,
            adapters={"data": adapter}
        )

        repr_str = repr(builder)

        assert "AsyncPromptBuilder" in repr_str
        assert "data" in repr_str


@pytest.mark.asyncio
class TestAsyncPromptBuilderIntegration:
    """Integration tests for AsyncPromptBuilder."""

    async def test_end_to_end_with_defaults_and_rag(self):
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

        class MockAsyncAdapter(AsyncDictResourceAdapter):
            async def search(self, query, k=5, filters=None, **kwargs):
                assert "python" in query.lower()
                return search_results

        builder = AsyncPromptBuilder(
            library=library,
            adapters={"docs": MockAsyncAdapter({})}
        )

        result = await builder.render_system_prompt(
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
