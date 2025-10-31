"""Unit tests for FileSystemPromptLibrary."""

import json
import pytest
import yaml
from pathlib import Path

from dataknobs_llm.prompts import (
    FileSystemPromptLibrary,
    ValidationLevel,
)


@pytest.fixture
def temp_prompts_dir(tmp_path):
    """Create a temporary prompts directory structure."""
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()

    # Create subdirectories
    (prompt_dir / "system").mkdir()
    (prompt_dir / "user").mkdir()
    (prompt_dir / "messages").mkdir()
    (prompt_dir / "rag").mkdir()

    return prompt_dir


class TestFileSystemLibraryInit:
    """Test suite for FileSystemPromptLibrary initialization."""

    def test_initialization_valid_directory(self, temp_prompts_dir):
        """Test initialization with valid directory."""
        library = FileSystemPromptLibrary(temp_prompts_dir, auto_load=False)
        assert library.prompt_dir == temp_prompts_dir

    def test_initialization_nonexistent_directory(self, tmp_path):
        """Test initialization with non-existent directory."""
        non_existent = tmp_path / "nonexistent"
        with pytest.raises(ValueError, match="does not exist"):
            FileSystemPromptLibrary(non_existent)

    def test_initialization_file_not_directory(self, tmp_path):
        """Test initialization with file instead of directory."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        with pytest.raises(ValueError, match="not a directory"):
            FileSystemPromptLibrary(file_path)

    def test_initialization_string_path(self, temp_prompts_dir):
        """Test initialization with string path."""
        library = FileSystemPromptLibrary(str(temp_prompts_dir), auto_load=False)
        assert library.prompt_dir == temp_prompts_dir

    def test_custom_file_extensions(self, temp_prompts_dir):
        """Test initialization with custom file extensions."""
        library = FileSystemPromptLibrary(
            temp_prompts_dir,
            auto_load=False,
            file_extensions=[".json"]
        )
        assert library.file_extensions == [".json"]


class TestFileSystemSystemPrompts:
    """Test suite for system prompts in FileSystemPromptLibrary."""

    def test_load_json_system_prompt(self, temp_prompts_dir):
        """Test loading system prompt from JSON file."""
        prompt_file = temp_prompts_dir / "system" / "analyze.json"
        prompt_data = {
            "template": "Analyze {{code}}",
            "defaults": {"language": "python"},
            "metadata": {"author": "test"}
        }
        prompt_file.write_text(json.dumps(prompt_data))

        library = FileSystemPromptLibrary(temp_prompts_dir)
        template = library.get_system_prompt("analyze")

        assert template is not None
        assert template["template"] == "Analyze {{code}}"
        assert template["defaults"]["language"] == "python"
        assert template["metadata"]["author"] == "test"

    def test_load_yaml_system_prompt(self, temp_prompts_dir):
        """Test loading system prompt from YAML file."""
        prompt_file = temp_prompts_dir / "system" / "greet.yaml"
        prompt_data = {
            "template": "Hello {{name}}!",
            "defaults": {"name": "World"}
        }
        prompt_file.write_text(yaml.dump(prompt_data))

        library = FileSystemPromptLibrary(temp_prompts_dir)
        template = library.get_system_prompt("greet")

        assert template is not None
        assert template["template"] == "Hello {{name}}!"

    def test_load_system_prompt_with_validation(self, temp_prompts_dir):
        """Test loading system prompt with validation config."""
        prompt_file = temp_prompts_dir / "system" / "validate.json"
        prompt_data = {
            "template": "Process {{input}}",
            "validation": {
                "level": "error",
                "required_params": ["input"]
            }
        }
        prompt_file.write_text(json.dumps(prompt_data))

        library = FileSystemPromptLibrary(temp_prompts_dir)
        template = library.get_system_prompt("validate")

        assert template is not None
        assert "validation" in template
        assert template["validation"].level == ValidationLevel.ERROR
        assert "input" in template["validation"].required_params

    def test_multiple_system_prompts(self, temp_prompts_dir):
        """Test loading multiple system prompts."""
        # Create multiple prompt files
        (temp_prompts_dir / "system" / "prompt1.json").write_text(
            json.dumps({"template": "Prompt 1"})
        )
        (temp_prompts_dir / "system" / "prompt2.json").write_text(
            json.dumps({"template": "Prompt 2"})
        )
        (temp_prompts_dir / "system" / "prompt3.json").write_text(
            json.dumps({"template": "Prompt 3"})
        )

        library = FileSystemPromptLibrary(temp_prompts_dir)

        assert library.get_system_prompt("prompt1") is not None
        assert library.get_system_prompt("prompt2") is not None
        assert library.get_system_prompt("prompt3") is not None

    def test_system_prompt_not_found(self, temp_prompts_dir):
        """Test retrieving non-existent system prompt."""
        library = FileSystemPromptLibrary(temp_prompts_dir)
        assert library.get_system_prompt("nonexistent") is None


class TestFileSystemUserPrompts:
    """Test suite for user prompts in FileSystemPromptLibrary."""

    def test_load_user_prompt_simple(self, temp_prompts_dir):
        """Test loading simple user prompt."""
        prompt_file = temp_prompts_dir / "user" / "question.json"
        prompt_data = {"template": "What is {{topic}}?"}
        prompt_file.write_text(json.dumps(prompt_data))

        library = FileSystemPromptLibrary(temp_prompts_dir)
        template = library.get_user_prompt("question")

        assert template is not None
        assert template["template"] == "What is {{topic}}?"

    def test_load_user_prompt_multiple_variants(self, temp_prompts_dir):
        """Test loading multiple user prompt variants."""
        # Create multiple named prompts
        prompt_file1 = temp_prompts_dir / "user" / "question.json"
        prompt_data1 = {"template": "Question about {{topic}}"}
        prompt_file1.write_text(json.dumps(prompt_data1))

        prompt_file2 = temp_prompts_dir / "user" / "followup.json"
        prompt_data2 = {"template": "Follow-up about {{topic}}"}
        prompt_file2.write_text(json.dumps(prompt_data2))

        prompt_file3 = temp_prompts_dir / "user" / "clarification.json"
        prompt_data3 = {"template": "Clarification about {{topic}}"}
        prompt_file3.write_text(json.dumps(prompt_data3))

        library = FileSystemPromptLibrary(temp_prompts_dir)

        template0 = library.get_user_prompt("question")
        template1 = library.get_user_prompt("followup")
        template2 = library.get_user_prompt("clarification")

        assert "Question about" in template0["template"]
        assert "Follow-up about" in template1["template"]
        assert "Clarification about" in template2["template"]

    def test_user_prompt_naming_format(self, temp_prompts_dir):
        """Test that user prompts can be any valid name."""
        # Valid formats
        valid_file1 = temp_prompts_dir / "user" / "valid.json"
        valid_file1.write_text(json.dumps({"template": "Valid"}))

        valid_file2 = temp_prompts_dir / "user" / "also_valid.json"
        valid_file2.write_text(json.dumps({"template": "Also valid"}))

        library = FileSystemPromptLibrary(temp_prompts_dir)

        # Both should be loaded
        assert library.get_user_prompt("valid") is not None
        assert library.get_user_prompt("also_valid") is not None

    def test_user_prompt_not_found(self, temp_prompts_dir):
        """Test retrieving non-existent user prompt."""
        library = FileSystemPromptLibrary(temp_prompts_dir)
        assert library.get_user_prompt("nonexistent") is None


class TestFileSystemMessageIndexes:
    """Test suite for message indexes in FileSystemPromptLibrary."""

    def test_load_message_index(self, temp_prompts_dir):
        """Test loading message index from file."""
        index_file = temp_prompts_dir / "messages" / "conversation.json"
        index_data = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ],
            "metadata": {"type": "greeting"}
        }
        index_file.write_text(json.dumps(index_data))

        library = FileSystemPromptLibrary(temp_prompts_dir)
        index = library.get_message_index("conversation")

        assert index is not None
        assert len(index["messages"]) == 2
        assert index["messages"][0]["role"] == "system"
        assert index["metadata"]["type"] == "greeting"

    def test_load_message_index_with_rag(self, temp_prompts_dir):
        """Test loading message index with RAG configs."""
        index_file = temp_prompts_dir / "messages" / "rag_chat.json"
        index_data = {
            "messages": [
                {"role": "system", "content": "Context: {{RAG_CONTENT}}"}
            ],
            "rag_configs": [
                {
                    "adapter_name": "docs",
                    "query": "{{user_query}}",
                    "k": 5
                }
            ]
        }
        index_file.write_text(json.dumps(index_data))

        library = FileSystemPromptLibrary(temp_prompts_dir)
        index = library.get_message_index("rag_chat")

        assert index is not None
        assert "rag_configs" in index
        assert index["rag_configs"][0]["adapter_name"] == "docs"

    def test_message_index_not_found(self, temp_prompts_dir):
        """Test retrieving non-existent message index."""
        library = FileSystemPromptLibrary(temp_prompts_dir)
        assert library.get_message_index("nonexistent") is None


class TestFileSystemRAGConfigs:
    """Test suite for RAG configurations in FileSystemPromptLibrary."""

    def test_load_rag_config(self, temp_prompts_dir):
        """Test loading RAG configuration from file."""
        rag_file = temp_prompts_dir / "rag" / "docs_search.json"
        rag_data = {
            "adapter_name": "documentation",
            "query": "{{user_query}}",
            "k": 3,
            "filters": {"category": "api"}
        }
        rag_file.write_text(json.dumps(rag_data))

        library = FileSystemPromptLibrary(temp_prompts_dir)
        rag = library.get_rag_config("docs_search")

        assert rag is not None
        assert rag["adapter_name"] == "documentation"
        assert rag["k"] == 3
        assert rag["filters"]["category"] == "api"

    def test_rag_config_with_custom_placeholder(self, temp_prompts_dir):
        """Test RAG config with custom placeholder and templates."""
        rag_file = temp_prompts_dir / "rag" / "custom.json"
        rag_data = {
            "adapter_name": "docs",
            "query": "search",
            "placeholder": "CUSTOM_CONTENT",
            "header": "Documentation:",
            "item_template": "- {{content}}"
        }
        rag_file.write_text(json.dumps(rag_data))

        library = FileSystemPromptLibrary(temp_prompts_dir)
        rag = library.get_rag_config("custom")

        assert rag is not None
        assert rag["placeholder"] == "CUSTOM_CONTENT"
        assert rag["header"] == "Documentation:"

    def test_rag_config_not_found(self, temp_prompts_dir):
        """Test retrieving non-existent RAG config."""
        library = FileSystemPromptLibrary(temp_prompts_dir)
        assert library.get_rag_config("nonexistent") is None


class TestFileSystemLibraryAutoLoad:
    """Test suite for auto-loading functionality."""

    def test_auto_load_true(self, temp_prompts_dir):
        """Test that auto_load=True loads prompts automatically."""
        # Create prompts
        (temp_prompts_dir / "system" / "test.json").write_text(
            json.dumps({"template": "Test"})
        )

        library = FileSystemPromptLibrary(temp_prompts_dir, auto_load=True)
        assert library.get_system_prompt("test") is not None

    def test_auto_load_false(self, temp_prompts_dir):
        """Test that auto_load=False does not load prompts."""
        # Create prompts
        (temp_prompts_dir / "system" / "test.json").write_text(
            json.dumps({"template": "Test"})
        )

        library = FileSystemPromptLibrary(temp_prompts_dir, auto_load=False)
        # Should not be loaded yet
        assert library.get_system_prompt("test") is None

        # Load manually
        library.load_all()
        assert library.get_system_prompt("test") is not None


class TestFileSystemLibraryEdgeCases:
    """Test edge cases for FileSystemPromptLibrary."""

    def test_missing_subdirectories(self, tmp_path):
        """Test library with missing subdirectories."""
        # Create directory with no subdirectories
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        # Should not raise error
        library = FileSystemPromptLibrary(empty_dir)

        assert library.get_system_prompt("any") is None
        assert library.get_user_prompt("any") is None

    def test_unsupported_file_extension(self, temp_prompts_dir):
        """Test that unsupported file extensions are ignored."""
        # Create file with unsupported extension
        (temp_prompts_dir / "system" / "test.txt").write_text("Not a prompt")

        library = FileSystemPromptLibrary(temp_prompts_dir)
        # Should not be loaded
        assert library.get_system_prompt("test") is None

    def test_malformed_json_file(self, temp_prompts_dir):
        """Test handling of malformed JSON file."""
        # Create malformed JSON
        (temp_prompts_dir / "system" / "bad.json").write_text("{ invalid json")

        # Should not raise error (logged as error)
        library = FileSystemPromptLibrary(temp_prompts_dir)
        assert library.get_system_prompt("bad") is None

    def test_empty_json_file(self, temp_prompts_dir):
        """Test handling of empty JSON file."""
        (temp_prompts_dir / "system" / "empty.json").write_text("{}")

        library = FileSystemPromptLibrary(temp_prompts_dir)
        template = library.get_system_prompt("empty")

        # Should load with empty template
        assert template is not None
        assert template["template"] == ""

    def test_mixed_file_types(self, temp_prompts_dir):
        """Test directory with both JSON and YAML files."""
        # Create JSON file
        (temp_prompts_dir / "system" / "json_prompt.json").write_text(
            json.dumps({"template": "JSON prompt"})
        )

        # Create YAML file (if PyYAML available)
        try:
            import yaml
            (temp_prompts_dir / "system" / "yaml_prompt.yaml").write_text(
                yaml.dump({"template": "YAML prompt"})
            )
        except ImportError:
            pass

        library = FileSystemPromptLibrary(temp_prompts_dir)

        # JSON should always be loaded
        assert library.get_system_prompt("json_prompt") is not None
