"""Filesystem-based prompt library implementation.

This module provides a prompt library that loads prompts from a directory structure
on the filesystem. Supports YAML and JSON formats.

Directory Structure:
    prompts/
        system/
            analyze_code.yaml
            review_pr.yaml
        user/
            code_question.yaml
            followup_question.yaml
        messages/
            conversation.yaml

File Format (YAML):
    template: |
        Analyze this {{language}} code:
        {{code}}
    defaults:
        language: python
    validation:
        level: error
        required_params:
            - code
    metadata:
        author: alice
        version: "1.0"
"""

import json
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Union

from ..base import (
    BasePromptLibrary,
    PromptTemplateDict,
    RAGConfig,
    MessageIndex,
)

logger = logging.getLogger(__name__)


class FileSystemPromptLibrary(BasePromptLibrary):
    """Prompt library that loads prompts from filesystem directory.

    Features:
    - Supports YAML and JSON file formats
    - Organized directory structure (system/, user/, messages/)
    - Caching of loaded prompts for performance
    - Automatic file discovery and loading
    - Validation config parsing from files

    Example:
        >>> library = FileSystemPromptLibrary(Path("prompts/"))
        >>> template = library.get_system_prompt("analyze_code")
        >>> print(template["template"])
    """

    def __init__(
        self,
        prompt_dir: Union[str, Path],
        auto_load: bool = True,
        file_extensions: List[str] | None = None
    ):
        """Initialize filesystem prompt library.

        Args:
            prompt_dir: Root directory containing prompt files
            auto_load: Whether to automatically load all prompts on init (default: True)
            file_extensions: List of file extensions to load (default: [".yaml", ".yml", ".json"])
        """
        super().__init__()

        self.prompt_dir = Path(prompt_dir)
        self.file_extensions = file_extensions or [".yaml", ".yml", ".json"]

        # Validate directory exists
        if not self.prompt_dir.exists():
            raise ValueError(f"Prompt directory does not exist: {self.prompt_dir}")

        if not self.prompt_dir.is_dir():
            raise ValueError(f"Prompt path is not a directory: {self.prompt_dir}")

        # Auto-load prompts if requested
        if auto_load:
            self.load_all()

    def load_all(self) -> None:
        """Load all prompts from the filesystem directory."""
        self._load_system_prompts()
        self._load_user_prompts()
        self._load_message_indexes()
        self._load_rag_configs()

    def _load_system_prompts(self) -> None:
        """Load all system prompts from system/ directory."""
        system_dir = self.prompt_dir / "system"
        if not system_dir.exists():
            logger.debug(f"System prompts directory not found: {system_dir}")
            return

        for file_path in system_dir.iterdir():
            if file_path.is_file() and file_path.suffix in self.file_extensions:
                name = file_path.stem
                try:
                    template = self._load_prompt_template(file_path)
                    self._cache_system_prompt(name, template)
                    logger.debug(f"Loaded system prompt: {name}")
                except Exception as e:
                    logger.error(f"Error loading system prompt {name}: {e}")

    def _load_user_prompts(self) -> None:
        """Load all user prompts from user/ directory.

        User prompts are loaded by name. Files should be named:
        - question.yaml
        - followup_question.yaml
        - etc.
        """
        user_dir = self.prompt_dir / "user"
        if not user_dir.exists():
            logger.debug(f"User prompts directory not found: {user_dir}")
            return

        for file_path in user_dir.iterdir():
            if file_path.is_file() and file_path.suffix in self.file_extensions:
                name = file_path.stem
                try:
                    template = self._load_prompt_template(file_path)
                    self._cache_user_prompt(name, template)
                    logger.debug(f"Loaded user prompt: {name}")
                except Exception as e:
                    logger.error(f"Error loading user prompt {name}: {e}")

    def _load_message_indexes(self) -> None:
        """Load all message indexes from messages/ directory."""
        messages_dir = self.prompt_dir / "messages"
        if not messages_dir.exists():
            logger.debug(f"Message indexes directory not found: {messages_dir}")
            return

        for file_path in messages_dir.iterdir():
            if file_path.is_file() and file_path.suffix in self.file_extensions:
                name = file_path.stem
                try:
                    message_index = self._load_message_index(file_path)
                    self._cache_message_index(name, message_index)
                    logger.debug(f"Loaded message index: {name}")
                except Exception as e:
                    logger.error(f"Error loading message index {name}: {e}")

    def _load_rag_configs(self) -> None:
        """Load all RAG configurations from rag/ directory."""
        rag_dir = self.prompt_dir / "rag"
        if not rag_dir.exists():
            logger.debug(f"RAG configs directory not found: {rag_dir}")
            return

        for file_path in rag_dir.iterdir():
            if file_path.is_file() and file_path.suffix in self.file_extensions:
                name = file_path.stem
                try:
                    rag_config = self._load_rag_config(file_path)
                    self._cache_rag_config(name, rag_config)
                    logger.debug(f"Loaded RAG config: {name}")
                except Exception as e:
                    logger.error(f"Error loading RAG config {name}: {e}")

    def _load_prompt_template(self, file_path: Path) -> PromptTemplateDict:
        """Load a prompt template from a file.

        Args:
            file_path: Path to the prompt template file

        Returns:
            PromptTemplateDict dictionary
        """
        data = self._load_file(file_path)

        # Use inherited _parse_prompt_template for consistent parsing
        # This supports templates with 'extends' but no 'template' field
        return self._parse_prompt_template(data)

    def _load_message_index(self, file_path: Path) -> MessageIndex:
        """Load a message index from a file.

        Args:
            file_path: Path to the message index file

        Returns:
            MessageIndex dictionary
        """
        data = self._load_file(file_path)

        # Build MessageIndex
        message_index: MessageIndex = {
            "messages": data.get("messages", []),
        }

        # Add optional fields
        if "rag_configs" in data:
            message_index["rag_configs"] = [
                self._parse_rag_config(rag_data)
                for rag_data in data["rag_configs"]
            ]

        if "metadata" in data:
            message_index["metadata"] = data["metadata"]

        return message_index

    def _load_rag_config(self, file_path: Path) -> RAGConfig:
        """Load a RAG configuration from a file.

        Args:
            file_path: Path to the RAG config file

        Returns:
            RAGConfig dictionary
        """
        data = self._load_file(file_path)
        return self._parse_rag_config(data)

    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse a YAML or JSON file.

        Args:
            file_path: Path to the file

        Returns:
            Parsed file contents as dictionary

        Raises:
            ValueError: If file format is unsupported or parsing fails
        """
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            if file_path.suffix in [".yaml", ".yml"]:
                return yaml.safe_load(content) or {}

            elif file_path.suffix == ".json":
                return json.loads(content)

            else:
                raise ValueError(f"Unsupported file extension: {file_path.suffix}")

        except Exception as e:
            raise ValueError(f"Error loading file {file_path}: {e}") from e

    # Note: _parse_prompt_template(), _parse_validation_config(), and
    # _parse_rag_config() are now inherited from BasePromptLibrary

    def get_system_prompt(self, name: str, **kwargs: Any) -> PromptTemplateDict | None:
        """Get a system prompt by name.

        Args:
            name: System prompt name
            **kwargs: Additional arguments (unused in filesystem library)

        Returns:
            PromptTemplateDict if found, None otherwise
        """
        return self._get_cached_system_prompt(name)

    def get_user_prompt(self, name: str, **kwargs: Any) -> PromptTemplateDict | None:
        """Get a user prompt by name.

        Args:
            name: User prompt name
            **kwargs: Additional arguments (unused in filesystem library)

        Returns:
            PromptTemplateDict if found, None otherwise
        """
        return self._get_cached_user_prompt(name)

    def get_message_index(self, name: str, **kwargs: Any) -> MessageIndex | None:
        """Get a message index by name.

        Args:
            name: Message index name
            **kwargs: Additional arguments (unused in filesystem library)

        Returns:
            MessageIndex if found, None otherwise
        """
        return self._get_cached_message_index(name)

    def get_rag_config(self, name: str, **kwargs: Any) -> RAGConfig | None:
        """Get a standalone RAG configuration by name.

        Args:
            name: RAG config name
            **kwargs: Additional arguments (unused in filesystem library)

        Returns:
            RAGConfig if found, None otherwise
        """
        return self._get_cached_rag_config(name)

    def get_prompt_rag_configs(
        self,
        prompt_name: str,
        prompt_type: str = "user",
        **kwargs: Any
    ) -> List[RAGConfig]:
        """Get RAG configurations for a specific prompt.

        Resolves both inline RAG configs and references to standalone configs.

        Args:
            prompt_name: Prompt name
            prompt_type: Type of prompt ("user" or "system")
            **kwargs: Additional arguments (unused)

        Returns:
            List of RAGConfig (empty if none defined)
        """
        # Get the prompt template
        if prompt_type == "system":
            template = self.get_system_prompt(prompt_name)
        else:
            template = self.get_user_prompt(prompt_name)

        if template is None:
            return []

        configs = []

        # Get inline RAG configs from template
        if "rag_configs" in template:
            configs.extend(template["rag_configs"])

        # Resolve RAG config references
        if "rag_config_refs" in template:
            for ref_name in template["rag_config_refs"]:
                ref_config = self.get_rag_config(ref_name)
                if ref_config:
                    configs.append(ref_config)
                else:
                    logger.warning(
                        f"RAG config reference '{ref_name}' not found "
                        f"for prompt '{prompt_name}'"
                    )

        return configs

    def list_system_prompts(self) -> List[str]:
        """List all available system prompt names.

        Returns:
            List of system prompt identifiers
        """
        return list(self._system_prompt_cache.keys())

    def list_user_prompts(self) -> List[str]:
        """List available user prompts.

        Returns:
            List of user prompt names
        """
        return list(self._user_prompt_cache.keys())

    def list_message_indexes(self) -> List[str]:
        """List all available message index names.

        Returns:
            List of message index identifiers
        """
        return list(self._message_index_cache.keys())
