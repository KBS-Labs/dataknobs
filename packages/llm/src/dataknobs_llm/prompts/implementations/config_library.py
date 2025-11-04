r"""Configuration-based prompt library implementation.

This module provides a prompt library that loads prompts from Python dictionaries.
Useful for programmatic prompt definition and testing.

Example:
    config = {
        "system": {
            "analyze_code": {
                "template": "Analyze this {{language}} code:\\n{{code}}",
                "defaults": {"language": "python"},
                "validation": {
                    "level": "error",
                    "required_params": ["code"]
                }
            }
        },
        "user": {
            "code_question": {
                "template": "Please analyze the following code...",
                "rag_configs": [...]
            },
            "followup_question": {
                "template": "Additionally, check for..."
            }
        }
    }

    library = ConfigPromptLibrary(config)
    template = library.get_system_prompt("analyze_code")
"""

import logging
from typing import Any, Dict, List

from ..base import (
    BasePromptLibrary,
    PromptTemplateDict,
    RAGConfig,
    MessageIndex,
)

logger = logging.getLogger(__name__)


class ConfigPromptLibrary(BasePromptLibrary):
    """Prompt library that loads prompts from configuration dictionaries.

    Features:
    - Direct in-memory configuration
    - No filesystem dependencies
    - Useful for testing and programmatic definition
    - Supports all prompt types (system, user, messages, RAG)

    Example:
        >>> config = {
        ...     "system": {"greet": {"template": "Hello {{name}}!"}},
        ...     "user": {"ask": {"template": "Tell me about {{topic}}"}}
        ... }
        >>> library = ConfigPromptLibrary(config)
        >>> template = library.get_system_prompt("greet")
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize configuration-based prompt library.

        Args:
            config: Configuration dictionary with structure:
                {
                    "system": {name: PromptTemplateDict, ...},
                    "user": {name: PromptTemplateDict, ...},
                    "messages": {name: MessageIndex, ...},
                    "rag": {name: RAGConfig, ...}
                }
        """
        super().__init__()
        self._config = config or {}

        # Load prompts from config
        self._load_from_config()

    def _load_from_config(self) -> None:
        """Load all prompts from the configuration dictionary."""
        self._load_system_prompts()
        self._load_user_prompts()
        self._load_message_indexes()
        self._load_rag_configs()  # Load standalone RAG configs

    def _load_system_prompts(self) -> None:
        """Load system prompts from config."""
        system_config = self._config.get("system", {})

        for name, data in system_config.items():
            try:
                template = self._parse_prompt_template(data)
                self._cache_system_prompt(name, template)
                logger.debug(f"Loaded system prompt from config: {name}")
            except Exception as e:
                logger.error(f"Error loading system prompt {name}: {e}")

    def _load_user_prompts(self) -> None:
        """Load user prompts from config."""
        user_config = self._config.get("user", {})

        for name, data in user_config.items():
            try:
                template = self._parse_prompt_template(data)
                self._cache_user_prompt(name, template)
                logger.debug(f"Loaded user prompt from config: {name}")
            except Exception as e:
                logger.error(f"Error loading user prompt {name}: {e}")

    def _load_message_indexes(self) -> None:
        """Load message indexes from config."""
        messages_config = self._config.get("messages", {})

        for name, data in messages_config.items():
            try:
                message_index = self._parse_message_index(data)
                self._cache_message_index(name, message_index)
                logger.debug(f"Loaded message index from config: {name}")
            except Exception as e:
                logger.error(f"Error loading message index {name}: {e}")

    def _load_rag_configs(self) -> None:
        """Load RAG configurations from config."""
        rag_config = self._config.get("rag", {})

        for name, data in rag_config.items():
            try:
                rag = self._parse_rag_config(data)
                self._cache_rag_config(name, rag)
                logger.debug(f"Loaded RAG config from config: {name}")
            except Exception as e:
                logger.error(f"Error loading RAG config {name}: {e}")

    # Note: _parse_prompt_template(), _parse_validation_config(), and
    # _parse_rag_config() are now inherited from BasePromptLibrary

    def _parse_message_index(self, data: Dict[str, Any]) -> MessageIndex:
        """Parse message index from config data.

        Args:
            data: Message index data

        Returns:
            MessageIndex dictionary
        """
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

    def get_system_prompt(self, name: str, **kwargs: Any) -> PromptTemplateDict | None:
        """Get a system prompt by name.

        Args:
            name: System prompt name
            **kwargs: Additional arguments (unused)

        Returns:
            PromptTemplateDict if found, None otherwise
        """
        return self._get_cached_system_prompt(name)

    def get_user_prompt(self, name: str, **kwargs: Any) -> PromptTemplateDict | None:
        """Get a user prompt by name.

        Args:
            name: User prompt name
            **kwargs: Additional arguments (unused)

        Returns:
            PromptTemplateDict if found, None otherwise
        """
        return self._get_cached_user_prompt(name)

    def get_message_index(self, name: str, **kwargs: Any) -> MessageIndex | None:
        """Get a message index by name.

        Args:
            name: Message index name
            **kwargs: Additional arguments (unused)

        Returns:
            MessageIndex if found, None otherwise
        """
        return self._get_cached_message_index(name)

    def get_rag_config(self, name: str, **kwargs: Any) -> RAGConfig | None:
        """Get a standalone RAG configuration by name.

        Args:
            name: RAG config name
            **kwargs: Additional arguments (unused)

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

    def add_system_prompt(self, name: str, template: PromptTemplateDict) -> None:
        """Add or update a system prompt.

        Args:
            name: System prompt name
            template: Prompt template to add
        """
        self._cache_system_prompt(name, template)
        logger.debug(f"Added/updated system prompt: {name}")

    def add_user_prompt(self, name: str, template: PromptTemplateDict) -> None:
        """Add or update a user prompt.

        Args:
            name: User prompt name
            template: Prompt template to add
        """
        self._cache_user_prompt(name, template)
        logger.debug(f"Added/updated user prompt: {name}")

    def add_message_index(self, name: str, message_index: MessageIndex) -> None:
        """Add or update a message index.

        Args:
            name: Message index name
            message_index: Message index to add
        """
        self._cache_message_index(name, message_index)
        logger.debug(f"Added/updated message index: {name}")

    def add_rag_config(self, name: str, rag_config: RAGConfig) -> None:
        """Add or update a RAG configuration.

        Args:
            name: RAG config name
            rag_config: RAG configuration to add
        """
        self._cache_rag_config(name, rag_config)
        logger.debug(f"Added/updated RAG config: {name}")

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
