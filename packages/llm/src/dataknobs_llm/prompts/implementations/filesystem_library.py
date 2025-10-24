"""Filesystem-based prompt library implementation.

This module provides a prompt library that loads prompts from a directory structure
on the filesystem. Supports YAML and JSON formats.

Directory Structure:
    prompts/
        system/
            analyze_code.yaml
            review_pr.yaml
        user/
            code_analysis_0.yaml
            code_analysis_1.yaml
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from ..base import (
    BasePromptLibrary,
    PromptTemplate,
    RAGConfig,
    MessageIndex,
    ValidationConfig,
    ValidationLevel,
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
        file_extensions: Optional[List[str]] = None
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

        # Check YAML support if needed
        if any(ext in [".yaml", ".yml"] for ext in self.file_extensions) and not HAS_YAML:
            logger.warning(
                "YAML file extensions specified but PyYAML not installed. "
                "Install with: pip install pyyaml"
            )

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

        User prompts are indexed by name and number. Files should be named:
        - prompt_name_0.yaml (index 0)
        - prompt_name_1.yaml (index 1)
        - etc.
        """
        user_dir = self.prompt_dir / "user"
        if not user_dir.exists():
            logger.debug(f"User prompts directory not found: {user_dir}")
            return

        for file_path in user_dir.iterdir():
            if file_path.is_file() and file_path.suffix in self.file_extensions:
                # Parse name and index from filename
                # Expected format: name_index.ext (e.g., code_analysis_0.yaml)
                stem = file_path.stem
                if '_' in stem:
                    parts = stem.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        name = parts[0]
                        index = int(parts[1])

                        try:
                            template = self._load_prompt_template(file_path)
                            self._cache_user_prompt(name, index, template)
                            logger.debug(f"Loaded user prompt: {name}[{index}]")
                        except Exception as e:
                            logger.error(f"Error loading user prompt {name}[{index}]: {e}")
                    else:
                        logger.warning(
                            f"User prompt file does not match expected format "
                            f"'name_index.ext': {file_path.name}"
                        )
                else:
                    logger.warning(
                        f"User prompt file does not match expected format "
                        f"'name_index.ext': {file_path.name}"
                    )

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

    def _load_prompt_template(self, file_path: Path) -> PromptTemplate:
        """Load a prompt template from a file.

        Args:
            file_path: Path to the prompt template file

        Returns:
            PromptTemplate dictionary
        """
        data = self._load_file(file_path)

        # Build PromptTemplate
        template: PromptTemplate = {
            "template": data.get("template", ""),
        }

        # Add optional fields if present
        if "defaults" in data:
            template["defaults"] = data["defaults"]

        if "validation" in data:
            template["validation"] = self._parse_validation_config(data["validation"])

        if "metadata" in data:
            template["metadata"] = data["metadata"]

        return template

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
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if file_path.suffix in [".yaml", ".yml"]:
                if not HAS_YAML:
                    raise ValueError(
                        f"PyYAML not installed, cannot load {file_path}. "
                        "Install with: pip install pyyaml"
                    )
                return yaml.safe_load(content) or {}

            elif file_path.suffix == ".json":
                return json.loads(content)

            else:
                raise ValueError(f"Unsupported file extension: {file_path.suffix}")

        except Exception as e:
            raise ValueError(f"Error loading file {file_path}: {e}")

    def _parse_validation_config(self, data: Union[Dict, ValidationConfig]) -> ValidationConfig:
        """Parse validation configuration from file data.

        Args:
            data: Validation data from file or ValidationConfig instance

        Returns:
            ValidationConfig instance
        """
        if isinstance(data, ValidationConfig):
            return data

        # Parse level
        level = None
        if "level" in data:
            level_str = data["level"].lower()
            level = ValidationLevel(level_str)

        # Parse params
        required_params = data.get("required_params", [])
        optional_params = data.get("optional_params", [])

        return ValidationConfig(
            level=level,
            required_params=required_params,
            optional_params=optional_params
        )

    def _parse_rag_config(self, data: Dict[str, Any]) -> RAGConfig:
        """Parse RAG configuration from file data.

        Args:
            data: RAG config data from file

        Returns:
            RAGConfig dictionary
        """
        rag_config: RAGConfig = {
            "adapter_name": data.get("adapter_name", ""),
            "query": data.get("query", ""),
        }

        # Add optional fields
        if "k" in data:
            rag_config["k"] = data["k"]

        if "filters" in data:
            rag_config["filters"] = data["filters"]

        if "placeholder" in data:
            rag_config["placeholder"] = data["placeholder"]

        if "header" in data:
            rag_config["header"] = data["header"]

        if "item_template" in data:
            rag_config["item_template"] = data["item_template"]

        return rag_config

    def get_system_prompt(self, name: str, **kwargs) -> Optional[PromptTemplate]:
        """Get a system prompt by name.

        Args:
            name: System prompt name
            **kwargs: Additional arguments (unused in filesystem library)

        Returns:
            PromptTemplate if found, None otherwise
        """
        return self._get_cached_system_prompt(name)

    def get_user_prompt(self, name: str, index: int = 0, **kwargs) -> Optional[PromptTemplate]:
        """Get a user prompt by name and index.

        Args:
            name: User prompt name
            index: Prompt index (default: 0)
            **kwargs: Additional arguments (unused in filesystem library)

        Returns:
            PromptTemplate if found, None otherwise
        """
        return self._get_cached_user_prompt(name, index)

    def get_message_index(self, name: str, **kwargs) -> Optional[MessageIndex]:
        """Get a message index by name.

        Args:
            name: Message index name
            **kwargs: Additional arguments (unused in filesystem library)

        Returns:
            MessageIndex if found, None otherwise
        """
        return self._get_cached_message_index(name)

    def get_rag_config(self, name: str, **kwargs) -> Optional[RAGConfig]:
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
        index: int = 0,
        **kwargs
    ) -> List[RAGConfig]:
        """Get RAG configurations for a specific prompt.

        Resolves both inline RAG configs and references to standalone configs.

        Args:
            prompt_name: Prompt name
            prompt_type: Type of prompt ("user" or "system")
            index: Prompt index (for user prompts)
            **kwargs: Additional arguments (unused)

        Returns:
            List of RAGConfig (empty if none defined)
        """
        # Get the prompt template
        if prompt_type == "system":
            template = self.get_system_prompt(prompt_name)
        else:
            template = self.get_user_prompt(prompt_name, index)

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

    def list_user_prompts(self, name: Optional[str] = None) -> List[str]:
        """List available user prompts.

        Args:
            name: If provided, list indices for this specific prompt

        Returns:
            List of user prompt names or indices
        """
        if name is None:
            # Return unique prompt names
            return list(set(key[0] for key in self._user_prompt_cache.keys()))
        else:
            # Return indices for this specific prompt
            indices = [
                str(key[1]) for key in self._user_prompt_cache.keys()
                if key[0] == name
            ]
            return sorted(indices, key=int)

    def list_message_indexes(self) -> List[str]:
        """List all available message index names.

        Returns:
            List of message index identifiers
        """
        return list(self._message_index_cache.keys())
