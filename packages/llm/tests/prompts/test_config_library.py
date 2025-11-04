"""Unit tests for ConfigPromptLibrary."""

import pytest
from dataknobs_llm.prompts import (
    ConfigPromptLibrary,
    ValidationLevel,
    ValidationConfig,
)


class TestConfigPromptLibrary:
    """Test suite for ConfigPromptLibrary basic functionality."""

    def test_initialization_empty(self):
        """Test initialization with empty config."""
        library = ConfigPromptLibrary()
        assert library.get_system_prompt("any") is None
        assert library.get_user_prompt("any") is None

    def test_initialization_with_config(self):
        """Test initialization with configuration."""
        config = {
            "system": {
                "greet": {"template": "Hello {{name}}!"}
            }
        }
        library = ConfigPromptLibrary(config)
        template = library.get_system_prompt("greet")
        assert template is not None
        assert template["template"] == "Hello {{name}}!"

    def test_system_prompt_simple(self):
        """Test loading simple system prompt."""
        config = {
            "system": {
                "analyze": {
                    "template": "Analyze {{code}}"
                }
            }
        }
        library = ConfigPromptLibrary(config)
        template = library.get_system_prompt("analyze")

        assert template is not None
        assert template["template"] == "Analyze {{code}}"

    def test_system_prompt_with_defaults(self):
        """Test system prompt with default values."""
        config = {
            "system": {
                "greet": {
                    "template": "Hello {{name}}!",
                    "defaults": {"name": "World"}
                }
            }
        }
        library = ConfigPromptLibrary(config)
        template = library.get_system_prompt("greet")

        assert template is not None
        assert template["defaults"]["name"] == "World"

    def test_system_prompt_with_validation(self):
        """Test system prompt with validation config."""
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
        template = library.get_system_prompt("analyze")

        assert template is not None
        assert "validation" in template
        assert template["validation"].level == ValidationLevel.ERROR
        assert "code" in template["validation"].required_params

    def test_system_prompt_with_metadata(self):
        """Test system prompt with metadata."""
        config = {
            "system": {
                "analyze": {
                    "template": "Analyze {{code}}",
                    "metadata": {
                        "author": "alice",
                        "version": "1.0"
                    }
                }
            }
        }
        library = ConfigPromptLibrary(config)
        template = library.get_system_prompt("analyze")

        assert template is not None
        assert template["metadata"]["author"] == "alice"
        assert template["metadata"]["version"] == "1.0"

    def test_system_prompt_as_string(self):
        """Test system prompt defined as plain string."""
        config = {
            "system": {
                "simple": "Just a template string"
            }
        }
        library = ConfigPromptLibrary(config)
        template = library.get_system_prompt("simple")

        assert template is not None
        assert template["template"] == "Just a template string"

    def test_system_prompt_not_found(self):
        """Test retrieving non-existent system prompt."""
        library = ConfigPromptLibrary({"system": {}})
        assert library.get_system_prompt("nonexistent") is None


class TestConfigUserPrompts:
    """Test suite for user prompts in ConfigPromptLibrary."""

    def test_user_prompt_simple(self):
        """Test simple user prompt."""
        config = {
            "user": {
                "question": {
                    "template": "What is {{topic}}?"
                }
            }
        }
        library = ConfigPromptLibrary(config)
        template = library.get_user_prompt("question")

        assert template is not None
        assert template["template"] == "What is {{topic}}?"

    def test_user_prompt_multiple_variants(self):
        """Test multiple user prompt variants."""
        config = {
            "user": {
                "question": {"template": "First question about {{topic}}"},
                "question_second": {"template": "Second question about {{topic}}"},
                "question_third": {"template": "Third question about {{topic}}"}
            }
        }
        library = ConfigPromptLibrary(config)

        template0 = library.get_user_prompt("question")
        template1 = library.get_user_prompt("question_second")
        template2 = library.get_user_prompt("question_third")

        assert "First question" in template0["template"]
        assert "Second question" in template1["template"]
        assert "Third question" in template2["template"]

    def test_user_prompt_multiple_names(self):
        """Test user prompts with different names."""
        config = {
            "user": {
                "question": {"template": "Question {{n}}"},
                "question_followup": {"template": "Question {{n}} follow-up"}
            }
        }
        library = ConfigPromptLibrary(config)

        template0 = library.get_user_prompt("question")
        template1 = library.get_user_prompt("question_followup")

        assert template0 is not None
        assert template1 is not None

    def test_user_prompt_with_defaults(self):
        """Test user prompt with default values."""
        config = {
            "user": {
                "ask": {
                    "template": "Tell me about {{topic}}",
                    "defaults": {"topic": "Python"}
                }
            }
        }
        library = ConfigPromptLibrary(config)
        template = library.get_user_prompt("ask")

        assert template["defaults"]["topic"] == "Python"

    def test_user_prompt_not_found(self):
        """Test retrieving non-existent user prompt."""
        library = ConfigPromptLibrary({"user": {}})
        assert library.get_user_prompt("nonexistent") is None


class TestConfigMessageIndexes:
    """Test suite for message indexes in ConfigPromptLibrary."""

    def test_message_index_simple(self):
        """Test simple message index."""
        config = {
            "messages": {
                "conversation": {
                    "messages": [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "Hello"}
                    ]
                }
            }
        }
        library = ConfigPromptLibrary(config)
        index = library.get_message_index("conversation")

        assert index is not None
        assert len(index["messages"]) == 2
        assert index["messages"][0]["role"] == "system"
        assert index["messages"][1]["role"] == "user"

    def test_message_index_with_rag(self):
        """Test message index with RAG configurations."""
        config = {
            "messages": {
                "rag_conversation": {
                    "messages": [
                        {"role": "system", "content": "Context: {{RAG_CONTENT}}"}
                    ],
                    "rag_configs": [
                        {
                            "adapter_name": "docs",
                            "query": "{{user_query}}",
                            "k": 5,
                            "placeholder": "RAG_CONTENT"
                        }
                    ]
                }
            }
        }
        library = ConfigPromptLibrary(config)
        index = library.get_message_index("rag_conversation")

        assert index is not None
        assert "rag_configs" in index
        assert len(index["rag_configs"]) == 1
        assert index["rag_configs"][0]["adapter_name"] == "docs"
        assert index["rag_configs"][0]["k"] == 5

    def test_message_index_with_metadata(self):
        """Test message index with metadata."""
        config = {
            "messages": {
                "chat": {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "metadata": {"type": "greeting"}
                }
            }
        }
        library = ConfigPromptLibrary(config)
        index = library.get_message_index("chat")

        assert index is not None
        assert index["metadata"]["type"] == "greeting"

    def test_message_index_not_found(self):
        """Test retrieving non-existent message index."""
        library = ConfigPromptLibrary({"messages": {}})
        assert library.get_message_index("nonexistent") is None


class TestConfigRAGConfigs:
    """Test suite for RAG configurations in ConfigPromptLibrary."""

    def test_rag_config_simple(self):
        """Test simple RAG configuration."""
        config = {
            "rag": {
                "docs_search": {
                    "adapter_name": "documentation",
                    "query": "{{user_query}}",
                    "k": 3
                }
            }
        }
        library = ConfigPromptLibrary(config)
        rag = library.get_rag_config("docs_search")

        assert rag is not None
        assert rag["adapter_name"] == "documentation"
        assert rag["query"] == "{{user_query}}"
        assert rag["k"] == 3

    def test_rag_config_with_filters(self):
        """Test RAG configuration with filters."""
        config = {
            "rag": {
                "filtered_search": {
                    "adapter_name": "docs",
                    "query": "search term",
                    "filters": {"category": "api", "version": "1.0"}
                }
            }
        }
        library = ConfigPromptLibrary(config)
        rag = library.get_rag_config("filtered_search")

        assert rag is not None
        assert rag["filters"]["category"] == "api"
        assert rag["filters"]["version"] == "1.0"

    def test_rag_config_with_placeholder(self):
        """Test RAG configuration with custom placeholder."""
        config = {
            "rag": {
                "custom_rag": {
                    "adapter_name": "docs",
                    "query": "query",
                    "placeholder": "CUSTOM_CONTENT",
                    "header": "Relevant Documentation:",
                    "item_template": "- {{content}}"
                }
            }
        }
        library = ConfigPromptLibrary(config)
        rag = library.get_rag_config("custom_rag")

        assert rag is not None
        assert rag["placeholder"] == "CUSTOM_CONTENT"
        assert rag["header"] == "Relevant Documentation:"
        assert rag["item_template"] == "- {{content}}"

    def test_rag_config_not_found(self):
        """Test retrieving non-existent RAG config."""
        library = ConfigPromptLibrary({"rag": {}})
        assert library.get_rag_config("nonexistent") is None


class TestConfigLibraryMutability:
    """Test suite for adding/updating prompts in ConfigPromptLibrary."""

    def test_add_system_prompt(self):
        """Test adding system prompt to library."""
        library = ConfigPromptLibrary()
        template = {"template": "New prompt {{x}}"}

        library.add_system_prompt("new_prompt", template)
        retrieved = library.get_system_prompt("new_prompt")

        assert retrieved is not None
        assert retrieved["template"] == "New prompt {{x}}"

    def test_update_system_prompt(self):
        """Test updating existing system prompt."""
        config = {"system": {"old": {"template": "Old"}}}
        library = ConfigPromptLibrary(config)

        new_template = {"template": "Updated"}
        library.add_system_prompt("old", new_template)

        retrieved = library.get_system_prompt("old")
        assert retrieved["template"] == "Updated"

    def test_add_user_prompt(self):
        """Test adding user prompt to library."""
        library = ConfigPromptLibrary()
        template = {"template": "User prompt {{y}}"}

        library.add_user_prompt("new_user", template)
        retrieved = library.get_user_prompt("new_user")

        assert retrieved is not None
        assert retrieved["template"] == "User prompt {{y}}"

    def test_add_message_index(self):
        """Test adding message index to library."""
        library = ConfigPromptLibrary()
        index = {
            "messages": [
                {"role": "user", "content": "Test"}
            ]
        }

        library.add_message_index("new_index", index)
        retrieved = library.get_message_index("new_index")

        assert retrieved is not None
        assert len(retrieved["messages"]) == 1

    def test_add_rag_config(self):
        """Test adding RAG config to library."""
        library = ConfigPromptLibrary()
        rag = {
            "adapter_name": "test",
            "query": "test query"
        }

        library.add_rag_config("new_rag", rag)
        retrieved = library.get_rag_config("new_rag")

        assert retrieved is not None
        assert retrieved["adapter_name"] == "test"


class TestConfigLibraryEdgeCases:
    """Test edge cases for ConfigPromptLibrary."""

    def test_empty_config(self):
        """Test with completely empty config."""
        library = ConfigPromptLibrary({})

        assert library.get_system_prompt("any") is None
        assert library.get_user_prompt("any") is None
        assert library.get_message_index("any") is None
        assert library.get_rag_config("any") is None

    def test_validation_level_variations(self):
        """Test different ways to specify validation level."""
        # String level
        config1 = {
            "system": {
                "test1": {
                    "template": "Test",
                    "validation": {"level": "warn"}
                }
            }
        }
        library1 = ConfigPromptLibrary(config1)
        template1 = library1.get_system_prompt("test1")
        assert template1["validation"].level == ValidationLevel.WARN

        # ValidationLevel enum
        config2 = {
            "system": {
                "test2": {
                    "template": "Test",
                    "validation": {"level": ValidationLevel.ERROR}
                }
            }
        }
        library2 = ConfigPromptLibrary(config2)
        template2 = library2.get_system_prompt("test2")
        assert template2["validation"].level == ValidationLevel.ERROR

    def test_validation_config_object(self):
        """Test passing ValidationConfig object directly."""
        validation = ValidationConfig(
            level=ValidationLevel.ERROR,
            required_params=["param1"]
        )
        config = {
            "system": {
                "test": {
                    "template": "Test {{param1}}",
                    "validation": validation
                }
            }
        }
        library = ConfigPromptLibrary(config)
        template = library.get_system_prompt("test")

        assert template["validation"] == validation
