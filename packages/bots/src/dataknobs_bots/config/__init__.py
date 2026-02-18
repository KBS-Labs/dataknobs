"""Configuration utilities for DynaBot.

This module provides utilities for resource resolution, configuration
binding, schema management, template-based building, draft management,
and validation.

Example:
    ```python
    from dataknobs_config import EnvironmentConfig
    from dataknobs_bots.config import (
        create_bot_resolver, BotResourceResolver,
        DynaBotConfigBuilder, DynaBotConfigSchema,
        ConfigValidator, ValidationResult,
        ConfigTemplate, ConfigTemplateRegistry, TemplateVariable,
        ConfigDraftManager, DraftMetadata,
    )

    # Build a config using the builder
    config = (
        DynaBotConfigBuilder()
        .set_llm("ollama", model="llama3.2")
        .set_conversation_storage("memory")
        .set_system_prompt(content="You are a helpful assistant.")
        .build()
    )

    # Or use templates
    registry = ConfigTemplateRegistry()
    registry.load_from_directory(Path("configs/templates"))
    config = registry.apply_template("basic_assistant", {"bot_name": "Helper"})
    ```
"""

from .builder import DynaBotConfigBuilder
from .drafts import ConfigDraftManager, DraftMetadata
from .wizard_builder import (
    StageConfig,
    TransitionConfig,
    WizardConfig,
    WizardConfigBuilder,
)
from .resolution import (
    BotResourceResolver,
    create_bot_resolver,
    register_database_factory,
    register_embedding_factory,
    register_llm_factory,
    register_vector_store_factory,
)
from .schema import DynaBotConfigSchema
from .templates import ConfigTemplate, ConfigTemplateRegistry, TemplateVariable
from .validation import ConfigValidator, ValidationResult
from .versioning import (
    ConfigVersion,
    ConfigVersionManager,
    VersionConflictError,
)

__all__ = [
    # Resolution
    "create_bot_resolver",
    "BotResourceResolver",
    "register_llm_factory",
    "register_database_factory",
    "register_vector_store_factory",
    "register_embedding_factory",
    # Versioning
    "ConfigVersion",
    "ConfigVersionManager",
    "VersionConflictError",
    # Config Toolkit
    "DynaBotConfigSchema",
    "ConfigValidator",
    "ValidationResult",
    "DynaBotConfigBuilder",
    "ConfigTemplate",
    "TemplateVariable",
    "ConfigTemplateRegistry",
    "ConfigDraftManager",
    "DraftMetadata",
    # Wizard config builder
    "WizardConfigBuilder",
    "WizardConfig",
    "StageConfig",
    "TransitionConfig",
]
