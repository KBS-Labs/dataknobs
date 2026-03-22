"""Core DynaBot implementation."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from dataknobs_llm import LLMStreamResponse
from dataknobs_llm.conversations import (
    ConversationManager,
    ConversationStorage,
    DataknobsConversationStorage,
)
from dataknobs_llm.conversations.storage import get_node_by_id, ConversationNode
from dataknobs_llm.llm import AsyncLLMProvider
from dataknobs_llm.prompts import AsyncPromptBuilder
from dataknobs_llm.tools import ToolRegistry
from dataknobs_llm.tools.context import ToolExecutionContext

from dataknobs_common.exceptions import NotFoundError

from ..knowledge.base import KnowledgeBase
from ..memory.base import Memory
from ..middleware.base import Middleware
from .context import BotContext
from .turn import ToolExecution, TurnMode, TurnState

if TYPE_CHECKING:
    from dataknobs_config import EnvironmentAwareConfig, EnvironmentConfig

logger = logging.getLogger(__name__)

# Provider role constants for the provider registry.
PROVIDER_ROLE_MAIN = "main"
PROVIDER_ROLE_EXTRACTION = "extraction"
PROVIDER_ROLE_MEMORY_EMBEDDING = "memory_embedding"
PROVIDER_ROLE_SUMMARY_LLM = "summary_llm"
PROVIDER_ROLE_KB_EMBEDDING = "kb_embedding"


def normalize_wizard_state(wizard_meta: dict[str, Any]) -> dict[str, Any]:
    """Normalize wizard metadata to canonical structure.

    Handles both old nested format (fsm_state.current_stage) and
    new flat format (current_stage directly).

    Args:
        wizard_meta: Raw wizard metadata from manager or storage

    Returns:
        Normalized wizard state dict with canonical fields:
        current_stage, stage_index, total_stages, progress, completed,
        data, can_skip, can_go_back, suggestions, history, stages,
        subflow_depth, and (when in a subflow) subflow_stage.
    """
    # Handle nested fsm_state format (legacy)
    fsm_state = wizard_meta.get("fsm_state", {})

    # Prefer direct fields, fall back to fsm_state
    current_stage = (
        wizard_meta.get("current_stage")
        or wizard_meta.get("stage")  # Old response format
        or fsm_state.get("current_stage")
    )

    result: dict[str, Any] = {
        "current_stage": current_stage,
        "stage_index": (
            wizard_meta.get("stage_index") or fsm_state.get("stage_index", 0)
        ),
        "total_stages": wizard_meta.get("total_stages", 0),
        "progress": wizard_meta.get("progress", 0.0),
        "completed": wizard_meta.get("completed", False),
        "data": wizard_meta.get("data") or fsm_state.get("data", {}),
        "can_skip": wizard_meta.get("can_skip", False),
        "can_go_back": wizard_meta.get("can_go_back", True),
        "suggestions": wizard_meta.get("suggestions", []),
        "history": wizard_meta.get("history") or fsm_state.get("history", []),
        "stages": wizard_meta.get("stages", []),
    }

    # Subflow context: present when wizard is executing a subflow
    subflow_stage = wizard_meta.get("subflow_stage")
    if subflow_stage:
        result["subflow_stage"] = subflow_stage
        result["subflow_depth"] = 1  # _build_wizard_metadata exposes top subflow
    else:
        result["subflow_depth"] = 0

    return result


@dataclass
class UndoResult:
    """Result of an undo operation."""

    undone_user_message: str
    undone_bot_response: str
    remaining_turns: int
    branching: bool


def _node_depth(node_id: str) -> int:
    """Depth of a node in the conversation tree. Root ("") is 0."""
    return len(node_id.split(".")) if node_id else 0


class DynaBot:
    """Configuration-driven chatbot leveraging the DataKnobs ecosystem.

    DynaBot provides a flexible, configuration-driven bot that can be customized
    for different use cases through YAML/JSON configuration files.

    .. versionadded:: 0.14.0
       DynaBot-level tool execution loop — strategies that pass tools to the
       LLM but do not execute ``tool_calls`` themselves (e.g. SimpleReasoning)
       now have their tool calls executed automatically by the bot pipeline.

    Attributes:
        llm: LLM provider for generating responses
        prompt_builder: Prompt builder for managing prompts
        conversation_storage: Storage backend for conversations
        tool_registry: Registry of available tools
        memory: Optional memory implementation for context
        knowledge_base: Optional knowledge base for RAG
        reasoning_strategy: Optional reasoning strategy
        middleware: List of middleware for request/response processing
        system_prompt_name: Name of the system prompt template to use
        system_prompt_content: Inline system prompt content (alternative to name)
        system_prompt_rag_configs: RAG configurations for inline system prompts
        default_temperature: Default temperature for LLM generation
        default_max_tokens: Default max tokens for LLM generation
    """

    _DEFAULT_MAX_TOOL_ITERATIONS = 5
    """Default maximum number of tool execution rounds before returning."""

    def __init__(
        self,
        llm: AsyncLLMProvider,
        prompt_builder: AsyncPromptBuilder,
        conversation_storage: ConversationStorage,
        tool_registry: ToolRegistry | None = None,
        memory: Memory | None = None,
        knowledge_base: KnowledgeBase | None = None,
        kb_auto_context: bool = True,
        reasoning_strategy: Any | None = None,
        middleware: list[Middleware] | None = None,
        system_prompt_name: str | None = None,
        system_prompt_content: str | None = None,
        system_prompt_rag_configs: list[dict[str, Any]] | None = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 1000,
        context_transform: Callable[[str], str] | None = None,
        max_tool_iterations: int = _DEFAULT_MAX_TOOL_ITERATIONS,
    ):
        """Initialize DynaBot.

        Args:
            llm: LLM provider instance
            prompt_builder: Prompt builder instance
            conversation_storage: Conversation storage backend
            tool_registry: Optional tool registry
            memory: Optional memory implementation
            knowledge_base: Optional knowledge base
            kb_auto_context: Whether to auto-inject KB results into messages.
                When False, the KB is still available for tool-based access
                but not automatically queried on every message.
            reasoning_strategy: Optional reasoning strategy
            middleware: Optional list of Middleware instances
            system_prompt_name: Name of system prompt template (mutually exclusive with content)
            system_prompt_content: Inline system prompt content (mutually exclusive with name)
            system_prompt_rag_configs: RAG configurations for inline system prompts
            default_temperature: Default temperature (0-1)
            default_max_tokens: Default max tokens to generate
            context_transform: Optional callable applied to each content string
                (KB chunks, memory context) before it is injected into the
                prompt.  Use this to sanitize or fence external content
                against prompt injection.
            max_tool_iterations: Maximum number of tool execution rounds
                before returning.  When a strategy returns a response with
                ``tool_calls``, DynaBot executes the tools and re-generates.
                This cap prevents infinite loops when the model keeps
                requesting the same tools.
        """
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.conversation_storage = conversation_storage
        self.tool_registry = tool_registry or ToolRegistry()
        self.memory = memory
        self.knowledge_base = knowledge_base
        self._kb_auto_context = kb_auto_context
        self.reasoning_strategy = reasoning_strategy
        self.middleware: list[Middleware] = middleware or []
        self.system_prompt_name = system_prompt_name
        self.system_prompt_content = system_prompt_content
        self.system_prompt_rag_configs = system_prompt_rag_configs
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self._context_transform = context_transform
        self._max_tool_iterations = max_tool_iterations
        self._owns_llm = True  # Set False by from_config() when llm= injected
        self._conversation_managers: dict[str, ConversationManager] = {}
        self._turn_checkpoints: dict[str, list[tuple[str, int]]] = {}
        self._providers: dict[str, AsyncLLMProvider] = {}

    def register_provider(self, role: str, provider: AsyncLLMProvider) -> None:
        """Register an auxiliary LLM/embedding provider by role.

        Providers registered here are included in ``all_providers`` for
        observability and enumeration.  The registry is a catalog — it
        does not manage provider lifecycle.  Each subsystem closes the
        providers it created (originator-owns-lifecycle).

        The ``"main"`` role is reserved for ``self.llm`` and cannot be
        overwritten.

        Args:
            role: Unique role identifier (e.g. ``"memory_embedding"``).
            provider: The provider instance.
        """
        if role == PROVIDER_ROLE_MAIN:
            logger.warning(
                "Cannot register provider with reserved role %r — "
                "use the 'llm' constructor parameter instead",
                PROVIDER_ROLE_MAIN,
            )
            return
        self._providers[role] = provider

    def get_provider(self, role: str) -> AsyncLLMProvider | None:
        """Get a registered provider by role.

        Args:
            role: Provider role identifier.

        Returns:
            The provider, or ``None`` if not registered.
        """
        if role == PROVIDER_ROLE_MAIN:
            return self.llm
        return self._providers.get(role)

    @property
    def all_providers(self) -> dict[str, AsyncLLMProvider]:
        """All registered providers keyed by role.

        Always includes ``"main"`` (``self.llm``).  Subsystems add
        their own entries during construction.  Returns a fresh dict
        (snapshot) on each call.
        """
        result: dict[str, AsyncLLMProvider] = {PROVIDER_ROLE_MAIN: self.llm}
        result.update(self._providers)
        return result

    @classmethod
    async def from_config(
        cls,
        config: dict[str, Any],
        *,
        llm: AsyncLLMProvider | None = None,
        middleware: list[Middleware] | None = None,
    ) -> DynaBot:
        """Create DynaBot from configuration.

        Args:
            config: Configuration dictionary containing:
                - llm: LLM configuration (provider, model, etc.).
                  Optional when the ``llm`` kwarg is provided.
                - conversation_storage: Storage configuration.  Two modes:
                    - ``backend``: Database backend key for the default
                      DataknobsConversationStorage (e.g. ``"memory"``,
                      ``"sqlite"``, ``"postgres"``).
                    - ``storage_class``: Dotted import path to a custom
                      ConversationStorage class (e.g.
                      ``"myapp.storage:AcmeStorage"``).  The class must
                      implement ``ConversationStorage`` including the
                      async ``create(config)`` classmethod.
                - tools: Optional list of tool configurations
                - memory: Optional memory configuration
                - knowledge_base: Optional knowledge base configuration
                - reasoning: Optional reasoning strategy configuration
                - middleware: Optional middleware configurations (ignored
                  when the ``middleware`` kwarg is provided)
                - prompts: Optional prompts library (dict of name -> content)
                - system_prompt: Optional system prompt configuration (see below)
                - config_base_path: Optional base directory for resolving
                  relative config file paths (e.g. wizard_config). When set,
                  relative paths in nested configs are resolved against this
                  directory instead of the current working directory.
            llm: Pre-built LLM provider.  When provided, ``config["llm"]``
                is optional and the provider is used as-is (no initialization
                or cleanup — the caller owns the lifecycle).  Use this to
                share a single provider across multiple bot instances.
            middleware: Pre-built middleware list.  When provided, replaces
                any middleware defined in config.

        Returns:
            Configured DynaBot instance

        System Prompt Formats:
            The system_prompt can be specified in multiple ways:

            - String: Smart detection - if the string exists as a template name
              in the prompt library, it's used as a template reference; otherwise
              it's treated as inline content.

            - Dict with name: `{"name": "template_name"}` - explicit template reference
            - Dict with name + strict: `{"name": "template_name", "strict": true}` -
              raises error if template doesn't exist
            - Dict with content: `{"content": "inline prompt text"}` - inline content
            - Dict with content + rag_configs: inline content with RAG enhancement

        Example:
            ```python
            bot = await DynaBot.from_config(config)

            # With a shared provider
            shared_llm = OllamaProvider({"provider": "ollama", "model": "llama3.2"})
            await shared_llm.initialize()
            bot = await DynaBot.from_config(
                {"conversation_storage": {"backend": "memory"}},
                llm=shared_llm,
            )

            # With pre-built middleware
            bot = await DynaBot.from_config(config, middleware=[my_middleware])
            ```
        """
        if llm is not None:
            # Caller-owned provider — skip creation/initialization.
            # Caller is responsible for lifecycle (initialize/close).
            llm_config = config.get("llm", {})
            bot = await cls._build_from_config(
                config, llm, llm_config, middleware_override=middleware
            )
            bot._owns_llm = False  # Caller owns lifecycle
            return bot

        # Create LLM provider from config
        llm_config = config["llm"]

        from dataknobs_llm.llm import LLMProviderFactory

        created_llm = LLMProviderFactory(is_async=True).create(llm_config)
        await created_llm.initialize()

        # Everything below can fail; ensure the provider is closed on error
        # so we don't leak aiohttp sessions or other resources.
        try:
            return await cls._build_from_config(
                config, created_llm, llm_config,
                middleware_override=middleware,
            )
        except Exception:
            await created_llm.close()
            raise

    @classmethod
    async def _build_from_config(
        cls,
        config: dict[str, Any],
        llm: Any,
        llm_config: dict[str, Any],
        *,
        middleware_override: list[Middleware] | None = None,
    ) -> DynaBot:
        """Build a DynaBot after the LLM provider is initialized.

        Separated from from_config() so the caller can guarantee cleanup
        of the LLM provider if anything here raises.
        """
        from dataknobs_llm.prompts import AsyncPromptBuilder
        from dataknobs_llm.prompts.implementations import CompositePromptLibrary

        from ..memory import create_memory_from_config

        # Validate capability requirements (Layer 2 — startup check)
        # Only check main LLM requirements here; extraction LLM requirements
        # are validated when WizardReasoning sets up its extractor.
        from .validation import infer_main_capability_requirements

        requirements = infer_main_capability_requirements(config)
        if requirements:
            capabilities = llm.get_capabilities()
            capability_values = {cap.value for cap in capabilities}
            missing = [r for r in requirements if r not in capability_values]
            if missing:
                from dataknobs_common.exceptions import ConfigurationError

                model_name = llm_config.get("model", "unknown")
                raise ConfigurationError(
                    f"Bot requires capabilities {missing} but model "
                    f"'{model_name}' provides "
                    f"{sorted(capability_values)}. "
                    f"Use a model that supports {missing} or "
                    f"update the environment resource configuration."
                )

        # Create conversation storage
        storage_config = config["conversation_storage"].copy()
        storage_class_path = storage_config.pop("storage_class", None)
        has_backend = "backend" in storage_config

        if storage_class_path and has_backend:
            logger.warning(
                "Both 'backend' and 'storage_class' specified in "
                "conversation_storage. 'storage_class' takes precedence; "
                "'backend' will be ignored."
            )
        if not storage_class_path and not has_backend:
            from dataknobs_common.exceptions import ConfigurationError

            raise ConfigurationError(
                "conversation_storage requires either 'backend' or "
                "'storage_class'. Use 'backend' for the default "
                "DataknobsConversationStorage, or 'storage_class' for a "
                "custom ConversationStorage implementation."
            )

        if storage_class_path:
            from dataknobs_bots.tools.resolve import resolve_callable

            storage_class = resolve_callable(storage_class_path)
            conversation_storage: ConversationStorage = await storage_class.create(
                storage_config
            )
        else:
            # Default: use DataknobsConversationStorage with database backend
            conversation_storage = await DataknobsConversationStorage.create(
                storage_config
            )

        # Create prompt builder
        # Support optional prompts configuration
        prompt_libraries = []
        if "prompts" in config:
            from dataknobs_llm.prompts.implementations import ConfigPromptLibrary

            prompts_config = config["prompts"]

            # If prompts are provided as a dict, create a config-based library
            if isinstance(prompts_config, dict):
                # Convert simple string prompts to proper template structure
                structured_config = {"system": {}, "user": {}}

                for prompt_name, prompt_content in prompts_config.items():
                    if isinstance(prompt_content, dict):
                        # Already structured - use as-is
                        # Assume it's a system prompt unless specified
                        prompt_type = prompt_content.get("type", "system")
                        if prompt_type in structured_config:
                            structured_config[prompt_type][prompt_name] = prompt_content
                    else:
                        # Simple string - treat as system prompt template
                        structured_config["system"][prompt_name] = {
                            "template": prompt_content
                        }

                library = ConfigPromptLibrary(structured_config)
                prompt_libraries.append(library)

        # Create composite library (empty if no prompts configured)
        library = CompositePromptLibrary(libraries=prompt_libraries)
        prompt_builder = AsyncPromptBuilder(library)

        # Create memory (pass llm so summary memory can use it)
        memory = None
        if "memory" in config:
            memory = await create_memory_from_config(
                config["memory"], llm_provider=llm
            )

        # Create knowledge base BEFORE tools — tools may declare a
        # dependency on knowledge_base via catalog_metadata().requires
        knowledge_base = None
        kb_config = config.get("knowledge_base", {})
        kb_auto_context = kb_config.get("auto_context", True)
        if kb_config.get("enabled"):
            from ..knowledge import create_knowledge_base_from_config

            logger.info("Initializing knowledge base with config: %s", kb_config.get("type", "unknown"))
            knowledge_base = await create_knowledge_base_from_config(kb_config)
            logger.info("Knowledge base initialized successfully")

        # Build dependency map for tool injection
        tool_dependencies: dict[str, Any] = {}
        if knowledge_base is not None:
            tool_dependencies["knowledge_base"] = knowledge_base

        # Create tools (after KB so dependencies can be injected)
        tool_registry = ToolRegistry()
        if "tools" in config:
            for tool_config in config["tools"]:
                tool = cls._resolve_tool(tool_config, config, tool_dependencies or None)
                if tool:
                    tool_registry.register_tool(tool)

        # Create reasoning strategy
        reasoning_strategy = None
        if "reasoning" in config:
            from ..reasoning import create_reasoning_from_config

            reasoning_config = config["reasoning"]
            # Propagate config_base_path to reasoning if set at bot level
            if "config_base_path" in config:
                if "config_base_path" not in reasoning_config:
                    reasoning_config = {
                        **reasoning_config,
                        "config_base_path": config["config_base_path"],
                    }
                elif reasoning_config["config_base_path"] != config["config_base_path"]:
                    logger.debug(
                        "Reasoning config has its own config_base_path=%r; "
                        "ignoring bot-level config_base_path=%r",
                        reasoning_config["config_base_path"],
                        config["config_base_path"],
                    )
            reasoning_strategy = create_reasoning_from_config(reasoning_config)

        # Create middleware
        if middleware_override is not None:
            middleware = list(middleware_override)
        else:
            middleware = []
            if "middleware" in config:
                for mw_config in config["middleware"]:
                    mw = cls._create_middleware(mw_config)
                    if mw:
                        middleware.append(mw)

        # Extract system prompt (supports template name or inline content)
        system_prompt_name = None
        system_prompt_content = None
        system_prompt_rag_configs = None
        if "system_prompt" in config:
            system_prompt_config = config["system_prompt"]
            if isinstance(system_prompt_config, dict):
                # Explicit dict format: {name: "template"} or {content: "inline..."}
                system_prompt_name = system_prompt_config.get("name")
                system_prompt_content = system_prompt_config.get("content")
                system_prompt_rag_configs = system_prompt_config.get("rag_configs")

                # If strict mode is enabled, require the template to exist
                if system_prompt_name and system_prompt_config.get("strict"):
                    if library.get_system_prompt(system_prompt_name) is None:
                        raise ValueError(
                            f"System prompt template not found: {system_prompt_name} "
                            "(strict mode enabled)"
                        )
            elif isinstance(system_prompt_config, str):
                # String format: smart detection
                # If it exists in the library, use as template name; otherwise treat as inline
                if library.get_system_prompt(system_prompt_config) is not None:
                    system_prompt_name = system_prompt_config
                else:
                    system_prompt_content = system_prompt_config

        # Resolve context_transform callable (if configured)
        context_transform: Callable[[str], str] | None = None
        context_transform_ref = config.get("context_transform")
        if context_transform_ref is not None:
            if callable(context_transform_ref):
                context_transform = context_transform_ref
            elif isinstance(context_transform_ref, str):
                from dataknobs_bots.tools.resolve import resolve_callable
                from dataknobs_common.exceptions import ConfigurationError

                try:
                    context_transform = resolve_callable(context_transform_ref)
                except (ImportError, AttributeError, ValueError) as exc:
                    raise ConfigurationError(
                        f"context_transform: could not resolve "
                        f"'{context_transform_ref}': {exc}"
                    ) from exc
            else:
                logger.warning(
                    "context_transform must be a callable or dotted import "
                    "string, got %s — ignoring",
                    type(context_transform_ref).__name__,
                )

        # Collect subsystem providers for catalog registration.
        # Each subsystem declares its own providers via providers().
        subsystem_providers: dict[str, AsyncLLMProvider] = {}

        if memory is not None:
            subsystem_providers.update(memory.providers())

        if knowledge_base is not None:
            subsystem_providers.update(knowledge_base.providers())

        if reasoning_strategy is not None:
            subsystem_providers.update(reasoning_strategy.providers())

        bot = cls(
            llm=llm,
            prompt_builder=prompt_builder,
            conversation_storage=conversation_storage,
            tool_registry=tool_registry,
            memory=memory,
            knowledge_base=knowledge_base,
            kb_auto_context=kb_auto_context,
            reasoning_strategy=reasoning_strategy,
            middleware=middleware,
            system_prompt_name=system_prompt_name,
            system_prompt_content=system_prompt_content,
            system_prompt_rag_configs=system_prompt_rag_configs,
            default_temperature=llm_config.get("temperature", 0.7),
            default_max_tokens=llm_config.get("max_tokens", 1000),
            context_transform=context_transform,
            max_tool_iterations=config.get(
                "max_tool_iterations", cls._DEFAULT_MAX_TOOL_ITERATIONS
            ),
        )

        for role, provider in subsystem_providers.items():
            bot.register_provider(role, provider)

        return bot

    @classmethod
    async def from_environment_aware_config(
        cls,
        config: EnvironmentAwareConfig | dict[str, Any],
        environment: EnvironmentConfig | str | None = None,
        env_dir: str | Path = "config/environments",
        config_key: str = "bot",
    ) -> DynaBot:
        """Create DynaBot with environment-aware configuration.

        This is the recommended entry point for environment-portable bots.
        Resource references ($resource) are resolved against the environment
        config, and environment variables are substituted at instantiation time
        (late binding).

        Args:
            config: EnvironmentAwareConfig instance or dict with $resource references.
                   If dict, will be wrapped in EnvironmentAwareConfig.
            environment: Environment name or EnvironmentConfig instance.
                        If None, auto-detects from DATAKNOBS_ENVIRONMENT env var.
                        Ignored if config is already an EnvironmentAwareConfig.
            env_dir: Directory containing environment config files.
                    Only used if environment is a string name.
            config_key: Key within config containing bot configuration.
                       Defaults to "bot". Set to None to use root config.

        Returns:
            Fully initialized DynaBot instance with resolved resources

        Example:
            ```python
            # With portable config dict
            config = {
                "bot": {
                    "llm": {
                        "$resource": "default",
                        "type": "llm_providers",
                        "temperature": 0.7,
                    },
                    "conversation_storage": {
                        "$resource": "conversations",
                        "type": "databases",
                    },
                }
            }
            bot = await DynaBot.from_environment_aware_config(config)

            # With explicit environment
            bot = await DynaBot.from_environment_aware_config(
                config,
                environment="production",
                env_dir="configs/environments"
            )

            # With EnvironmentAwareConfig instance
            from dataknobs_config import EnvironmentAwareConfig
            env_config = EnvironmentAwareConfig.load_app("my-bot", ...)
            bot = await DynaBot.from_environment_aware_config(env_config)
            ```

        Note:
            The config should use $resource references for infrastructure:
            ```yaml
            bot:
              llm:
                $resource: default      # Logical name
                type: llm_providers     # Resource type
                temperature: 0.7        # Behavioral param (portable)
            ```

            The environment config provides concrete bindings:
            ```yaml
            resources:
              llm_providers:
                default:
                  provider: openai
                  model: gpt-4
                  api_key: ${OPENAI_API_KEY}
            ```
        """
        from dataknobs_config import EnvironmentAwareConfig, EnvironmentConfig

        # Wrap dict in EnvironmentAwareConfig if needed
        if isinstance(config, dict):
            # Load or use provided environment
            if isinstance(environment, EnvironmentConfig):
                env_config = environment
            else:
                env_config = EnvironmentConfig.load(environment, env_dir)

            config = EnvironmentAwareConfig(
                config=config,
                environment=env_config,
            )
        elif environment is not None:
            # Switch environment on existing EnvironmentAwareConfig
            config = config.with_environment(environment, env_dir)

        # Resolve resources and env vars (late binding happens here)
        if config_key:
            resolved = config.resolve_for_build(config_key)
        else:
            resolved = config.resolve_for_build()

        # Delegate to existing from_config
        return await cls.from_config(resolved)

    @staticmethod
    def get_portable_config(
        config: EnvironmentAwareConfig | dict[str, Any],
    ) -> dict[str, Any]:
        """Extract portable configuration for storage.

        Returns configuration with $resource references intact
        and environment variables unresolved. This is the config
        that should be stored in registries or databases for
        cross-environment portability.

        Args:
            config: EnvironmentAwareConfig instance or portable dict

        Returns:
            Portable configuration dictionary

        Example:
            ```python
            from dataknobs_config import EnvironmentAwareConfig

            # From EnvironmentAwareConfig
            env_config = EnvironmentAwareConfig.load_app("my-bot", ...)
            portable = DynaBot.get_portable_config(env_config)

            # Store portable config in registry
            await registry.store(bot_id, portable)

            # Dict passes through unchanged
            portable = DynaBot.get_portable_config({"bot": {...}})
            ```
        """
        # Import here to avoid circular dependency at module level
        try:
            from dataknobs_config import EnvironmentAwareConfig

            if isinstance(config, EnvironmentAwareConfig):
                return config.get_portable_config()
        except ImportError:
            pass

        # Dict passes through (assumed already portable)
        return config

    async def _prepare_turn(self, turn: TurnState) -> None:
        """Shared pre-processing for all turn types.

        For chat/stream: runs on_turn_start (plugin_data + message
        transforms), before_message middleware, builds the augmented
        message, gets/creates the conversation manager, records undo
        checkpoint, adds the user message, updates memory, and injects
        plugin_data into the manager for LLM middleware access.

        For greet: runs on_turn_start and before_message middleware
        (empty message) and gets/creates the conversation manager.
        No user message is added.

        Args:
            turn: Turn state to populate with the conversation manager.
        """
        # on_turn_start: plugin_data writes + message transforms (chained)
        await self._call_on_turn_start_middleware(turn)

        # Legacy observational hook
        await self._call_before_message_middleware(turn.message, turn.context)

        if turn.is_greet:
            turn.manager = await self._get_or_create_conversation(turn.context)
            # Bridge plugin_data to LLM middleware
            if turn.manager.state is not None:
                turn.manager.state.turn_data = turn.plugin_data
            return

        # Build message with context from memory and knowledge
        full_message = await self._build_message_with_context(
            turn.message, rag_query=turn.rag_query
        )

        # Get or create conversation manager
        turn.manager = await self._get_or_create_conversation(turn.context)

        # Record tree position before the turn for undo support.
        # Store (node_id, memory_count) — node_id for tree navigation,
        # memory_count for accurate memory rollback (node depth is unreliable).
        conv_id = turn.context.conversation_id
        if conv_id not in self._turn_checkpoints:
            self._turn_checkpoints[conv_id] = []
        mem_count = 0
        if self.memory:
            try:
                mem_count = len(await self.memory.get_context(""))
            except Exception:
                mem_count = 0
        self._turn_checkpoints[conv_id].append((
            turn.manager.state.current_node_id if turn.manager.state else "",
            mem_count,
        ))

        # Add user message.  When context augmentation was applied (KB
        # results, memory history), store the original raw message in node
        # metadata so that downstream consumers (e.g. WizardReasoning
        # extraction) can access the undecorated user input.
        msg_metadata: dict[str, Any] | None = None
        if full_message != turn.message:
            msg_metadata = {"raw_content": turn.message}
        await turn.manager.add_message(
            content=full_message, role="user", metadata=msg_metadata
        )

        # Update memory
        if self.memory:
            await self.memory.add_message(turn.message, role="user")

        # Bridge plugin_data to LLM middleware via ConversationState.turn_data
        if turn.manager.state is not None:
            turn.manager.state.turn_data = turn.plugin_data

    async def _execute_tools(
        self,
        turn: TurnState,
        tool_calls: list[Any],
    ) -> None:
        """Execute tool calls and add observations to the conversation.

        Builds a ``ToolExecutionContext`` from the conversation manager,
        executes each tool, records ``ToolExecution`` on the turn state,
        and adds tool result observations to the conversation history so
        the next LLM call sees them.

        Args:
            turn: Current turn state (tool executions are appended here).
            tool_calls: List of ``ToolCall`` objects from the LLM response.
        """
        for tool_call in tool_calls:
            tool_name = tool_call.name
            tool_context = ToolExecutionContext.from_manager(turn.manager)
            try:
                tool = self.tool_registry.get_tool(tool_name)
            except NotFoundError:
                tool = None

            if tool is None:
                observation = "Tool not found"
                turn.tool_executions.append(ToolExecution(
                    tool_name=tool_name,
                    parameters=tool_call.parameters,
                    error="Tool not found",
                ))
                logger.warning(
                    "Tool not found: %s",
                    tool_name,
                    extra={
                        "conversation_id": getattr(
                            turn.manager, "conversation_id", None
                        ),
                    },
                )
            else:
                try:
                    t0 = time.monotonic()
                    result = await tool.execute(
                        **tool_call.parameters, _context=tool_context
                    )
                    duration_ms = (time.monotonic() - t0) * 1000
                    try:
                        observation = (
                            f"Tool result: {json.dumps(result, default=str)}"
                        )
                    except (TypeError, ValueError):
                        observation = f"Tool result: {result}"

                    turn.tool_executions.append(ToolExecution(
                        tool_name=tool_name,
                        parameters=tool_call.parameters,
                        result=result,
                        duration_ms=duration_ms,
                    ))
                    logger.info(
                        "Tool executed: %s",
                        tool_name,
                        extra={
                            "conversation_id": getattr(
                                turn.manager, "conversation_id", None
                            ),
                            "duration_ms": round(duration_ms, 1),
                            "result_length": len(str(result)),
                        },
                    )
                except Exception as exc:
                    observation = f"Error: {exc!s}"
                    turn.tool_executions.append(ToolExecution(
                        tool_name=tool_name,
                        parameters=tool_call.parameters,
                        error=str(exc),
                    ))
                    logger.error(
                        "Tool execution failed: %s",
                        tool_name,
                        extra={
                            "conversation_id": getattr(
                                turn.manager, "conversation_id", None
                            ),
                            "error": str(exc),
                        },
                        exc_info=True,
                    )

            await turn.manager.add_message(
                content=f"Observation from {tool_name}: {observation}",
                role="tool",
                name=tool_name,
            )

    async def _finalize_turn(self, turn: TurnState) -> None:
        """Shared post-generation processing for all turn types.

        Updates memory with the assistant response, fires tool execution
        hooks, dispatches the unified ``after_turn`` middleware hook, and
        then dispatches the appropriate legacy hook (``after_message`` for
        chat/greet, ``post_stream`` for streaming).

        Args:
            turn: Completed turn state with response content populated.
        """
        # Update memory with assistant response
        if self.memory and turn.response_content:
            await self.memory.add_message(turn.response_content, role="assistant")

        # Collect tool executions from strategy (appended after DynaBot-level
        # executions — ordering is by source, not chronological).  For current
        # strategies, only one source produces executions per turn: either
        # ReAct (strategy-level) or the DynaBot loop (for non-ReAct strategies).
        if self.reasoning_strategy:
            strategy_tools = self.reasoning_strategy.get_and_clear_tool_executions()
            turn.tool_executions.extend(strategy_tools)

        # Fire on_tool_executed for each tool execution (post-turn, not
        # real-time — middleware cannot abort or rate-limit mid-turn).
        for execution in turn.tool_executions:
            await self._call_on_tool_executed_middleware(execution, turn.context)

        # New unified hook — all turn types
        await self._call_after_turn_middleware(turn)

        # Legacy hooks for backward compatibility
        if turn.is_streaming:
            await self._call_post_stream_middleware(
                turn.message, turn.response_content, turn.context
            )
        else:
            mw_kwargs = turn.middleware_kwargs()
            await self._call_after_message_middleware(
                turn.response_content, turn.context, **mw_kwargs
            )

        # Clean up transient turn_data on the manager to avoid leaking
        # between turns (the manager is cached across turns).
        if turn.manager and turn.manager.state is not None:
            turn.manager.state.turn_data = {}

    async def _generate_response(
        self,
        manager: Any,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config_overrides: dict[str, Any] | None = None,
    ) -> Any:
        """Dispatch response generation through reasoning strategy or direct completion.

        Args:
            manager: ConversationManager instance
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            llm_config_overrides: Optional per-request LLM config overrides

        Returns:
            LLM response object.
        """
        if self.reasoning_strategy:
            return await self.reasoning_strategy.generate(
                manager=manager,
                llm=self.llm,
                tools=list(self.tool_registry) or None,
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
                llm_config_overrides=llm_config_overrides,
            )
        return await manager.complete(
            tools=list(self.tool_registry) or None,
            llm_config_overrides=llm_config_overrides,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
        )

    @staticmethod
    def _extract_response_content(response: Any) -> str:
        """Extract text content from an LLM response object.

        Args:
            response: LLM response (may have .content attribute or be a string)

        Returns:
            The response text as a string.
        """
        return response.content if hasattr(response, "content") else str(response)

    async def _call_on_turn_start_middleware(self, turn: TurnState) -> None:
        """Dispatch on_turn_start to all middleware (chained transforms).

        Each middleware can write to ``turn.plugin_data`` and optionally
        return a transformed message. Transforms chain: each middleware
        receives the message as modified by the previous one.

        Every middleware gets called even if an earlier one raises.
        If any raise, the first error is re-raised after all have
        been called (so the outer try block can route it to on_error).
        This matches ``before_message`` semantics — middleware can
        raise to abort the request (e.g. rate limiting, auth).

        Args:
            turn: Turn state at the start of the pipeline.
        """
        first_error: Exception | None = None
        for mw in self.middleware:
            try:
                result = await mw.on_turn_start(turn)
                if result is not None:
                    turn.message = result
            except Exception as exc:
                logger.exception(
                    "Middleware %s.on_turn_start raised",
                    type(mw).__name__,
                )
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    async def _call_before_message_middleware(
        self, message: str, context: BotContext
    ) -> None:
        """Dispatch before_message to all middleware.

        Every middleware gets called even if an earlier one raises.
        If any raise, the first error is re-raised after all have
        been called (so the outer try block can route it to on_error).

        Args:
            message: User message (empty string for greet)
            context: Bot execution context
        """
        first_error: Exception | None = None
        for mw in self.middleware:
            try:
                await mw.before_message(message, context)
            except Exception as exc:
                logger.exception(
                    "Middleware %s.before_message raised",
                    type(mw).__name__,
                )
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    async def _call_after_message_middleware(
        self,
        response_content: str,
        context: BotContext,
        **kwargs: Any,
    ) -> None:
        """Dispatch after_message to all middleware.

        Observational hook — one failing middleware must not prevent
        others from being notified. Errors are logged, then reported
        to all middleware via ``on_hook_error``.

        Args:
            response_content: Bot response text
            context: Bot execution context
            **kwargs: Additional data (tokens_used, model, provider)
        """
        for mw in self.middleware:
            try:
                await mw.after_message(response_content, context, **kwargs)
            except Exception as exc:
                logger.exception(
                    "Middleware %s.after_message raised",
                    type(mw).__name__,
                )
                await self._call_on_hook_error_middleware(
                    "after_message", exc, context
                )

    async def _call_post_stream_middleware(
        self, message: str, response: str, context: BotContext
    ) -> None:
        """Dispatch post_stream to all middleware.

        Observational hook — one failing middleware must not prevent
        others from being notified. Errors are logged, then reported
        to all middleware via ``on_hook_error``.

        Args:
            message: Original user message
            response: Complete accumulated response
            context: Bot execution context
        """
        for mw in self.middleware:
            try:
                await mw.post_stream(message, response, context)
            except Exception as exc:
                logger.exception(
                    "Middleware %s.post_stream raised",
                    type(mw).__name__,
                )
                await self._call_on_hook_error_middleware(
                    "post_stream", exc, context
                )

    async def _call_on_error_middleware(
        self, error: Exception, message: str, context: BotContext
    ) -> None:
        """Dispatch on_error to all middleware.

        Error notification hook — one failing middleware must not prevent
        others from being notified. Errors are logged, then reported
        to all middleware via ``on_hook_error``.

        Called by chat(), greet(), and stream_chat() when an exception
        occurs during message processing. The caller re-raises the
        original exception after this method returns.

        Args:
            error: The exception that occurred
            message: User message that triggered the error (empty string
                for greet, which has no user message)
            context: Bot execution context
        """
        for mw in self.middleware:
            try:
                await mw.on_error(error, message, context)
            except Exception as exc:
                logger.exception(
                    "Middleware %s.on_error raised during error dispatch",
                    type(mw).__name__,
                )
                await self._call_on_hook_error_middleware(
                    "on_error", exc, context
                )

    async def _call_on_hook_error_middleware(
        self, hook_name: str, error: Exception, context: BotContext
    ) -> None:
        """Dispatch on_hook_error to all middleware.

        Called when a middleware hook itself fails. Unlike ``on_error``,
        this does NOT mean the request failed — it means a middleware
        could not complete its own post-processing.

        All middleware receive the notification, including the middleware
        whose hook failed — it sees its own failure reported back via
        ``on_hook_error``.  Each call is independent: if ``on_hook_error``
        itself raises, the failure is logged but not re-dispatched
        (no infinite recursion).

        Args:
            hook_name: Name of the hook that failed (e.g. "after_message")
            error: The exception raised by the middleware hook
            context: Bot execution context
        """
        for mw in self.middleware:
            try:
                await mw.on_hook_error(hook_name, error, context)
            except Exception:
                logger.exception(
                    "Middleware %s.on_hook_error raised (hook: %s)",
                    type(mw).__name__,
                    hook_name,
                )

    async def _call_after_turn_middleware(self, turn: TurnState) -> None:
        """Dispatch after_turn to all middleware.

        Observational hook — one failing middleware must not prevent
        others from being notified. Errors are logged, then reported
        to all middleware via ``on_hook_error``.

        Args:
            turn: Completed turn state.
        """
        for mw in self.middleware:
            try:
                await mw.after_turn(turn)
            except Exception as exc:
                logger.exception(
                    "Middleware %s.after_turn raised",
                    type(mw).__name__,
                )
                await self._call_on_hook_error_middleware(
                    "after_turn", exc, turn.context
                )

    async def _call_on_tool_executed_middleware(
        self, execution: ToolExecution, context: BotContext
    ) -> None:
        """Dispatch on_tool_executed to all middleware.

        Observational hook — one failing middleware must not prevent
        others from being notified. Errors are logged, then reported
        to all middleware via ``on_hook_error``.

        Args:
            execution: Record of the tool execution.
            context: Bot execution context.
        """
        for mw in self.middleware:
            try:
                await mw.on_tool_executed(execution, context)
            except Exception as exc:
                logger.exception(
                    "Middleware %s.on_tool_executed raised",
                    type(mw).__name__,
                )
                await self._call_on_hook_error_middleware(
                    "on_tool_executed", exc, context
                )

    async def chat(
        self,
        message: str,
        context: BotContext,
        temperature: float | None = None,
        max_tokens: int | None = None,
        rag_query: str | None = None,
        llm_config_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Process a chat message.

        Args:
            message: User message to process
            context: Bot execution context
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            rag_query: Optional explicit query for knowledge base retrieval.
                      If provided, this is used instead of the message for RAG.
                      Useful when the message contains literal text to analyze
                      (e.g., "Analyze this prompt: [prompt text]") but you want
                      to search for analysis techniques instead.
            llm_config_overrides: Optional dict to override LLM config fields
                      for this request only. Supported fields: model, temperature,
                      max_tokens, top_p, stop_sequences, seed, options.
            **kwargs: Additional arguments

        Returns:
            Bot response as string

        Example:
            ```python
            context = BotContext(
                conversation_id="conv-123",
                client_id="client-456",
                user_id="user-789"
            )
            response = await bot.chat("Hello!", context)

            # With explicit RAG query
            response = await bot.chat(
                "Analyze this: Write a poem about cats",
                context,
                rag_query="prompt analysis techniques evaluation"
            )

            # With LLM config overrides (switch model per-request)
            response = await bot.chat(
                "Explain quantum computing",
                context,
                llm_config_overrides={"model": "gpt-4-turbo", "temperature": 0.9}
            )
            ```
        """
        turn = TurnState(
            mode=TurnMode.CHAT,
            message=message,
            context=context,
            rag_query=rag_query,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_config_overrides=llm_config_overrides,
        )
        try:
            await self._prepare_turn(turn)
            response = await self._generate_response(
                turn.manager, temperature, max_tokens, llm_config_overrides
            )

            # DynaBot-level tool execution loop.  Strategies that handle
            # tool_calls internally (e.g. ReAct) return responses without
            # tool_calls, so this loop is a no-op for them.
            for _iteration in range(self._max_tool_iterations):
                if (
                    not self.tool_registry
                    or not getattr(response, "tool_calls", None)
                ):
                    break
                await self._execute_tools(turn, response.tool_calls)
                # Accumulate usage from intermediate LLM calls
                turn.accumulate_usage(response)
                response = await turn.manager.complete(
                    tools=list(self.tool_registry) or None,
                    temperature=temperature or self.default_temperature,
                    max_tokens=max_tokens or self.default_max_tokens,
                    llm_config_overrides=llm_config_overrides,
                )
            else:
                # Loop completed without break — cap hit
                if self.tool_registry and getattr(
                    response, "tool_calls", None
                ):
                    logger.warning(
                        "Tool execution loop reached max iterations (%d) "
                        "with pending tool_calls",
                        self._max_tool_iterations,
                        extra={
                            "conversation_id": getattr(
                                turn.manager, "conversation_id", None
                            ),
                        },
                    )

            turn.response = response
            turn.response_content = self._extract_response_content(response)
            turn.populate_from_response(response, self.llm)
            await self._finalize_turn(turn)
            return turn.response_content
        except Exception as e:
            await self._call_on_error_middleware(e, message, context)
            raise

    async def greet(
        self,
        context: BotContext,
        *,
        initial_context: dict[str, Any] | None = None,
    ) -> str | None:
        """Generate a bot-initiated greeting before the user speaks.

        Delegates to the reasoning strategy's ``greet()`` method. Returns
        ``None`` if the bot has no reasoning strategy or the strategy does
        not support greetings (e.g. non-wizard strategies).

        No user message is added to conversation history — the greeting
        is a bot-initiated assistant message only.

        Args:
            context: Bot execution context
            initial_context: Optional dict of initial data to seed into
                the reasoning strategy's state before generating the
                greeting. For wizard strategies, these values are merged
                into ``wizard_state.data`` so they are available to the
                start stage's prompt template and transforms.

        Returns:
            Greeting string, or None if the bot does not support greetings

        Note:
            Middleware lifecycle for greet: ``on_turn_start(turn)`` and
            ``before_message("")`` are called before greeting generation;
            ``after_turn(turn)`` and ``after_message(...)`` are called on
            success.  If an error occurs, ``on_error`` hooks receive
            ``message=""`` since there is no user message.  If a
            middleware hook itself fails, ``on_hook_error`` is called on
            all middleware.

        Example:
            ```python
            context = BotContext(conversation_id="conv-123", client_id="harness")
            greeting = await bot.greet(context, initial_context={"user_name": "Alice"})
            if greeting:
                print(f"Bot says: {greeting}")
            ```
        """
        if not self.reasoning_strategy:
            return None

        turn = TurnState(
            mode=TurnMode.GREET,
            message="",
            context=context,
            initial_context=initial_context,
        )
        try:
            await self._prepare_turn(turn)

            response = await self.reasoning_strategy.greet(
                manager=turn.manager,
                llm=self.llm,
                initial_context=initial_context,
            )

            if response is None:
                return None

            turn.response = response
            turn.response_content = self._extract_response_content(response)
            # Note: greet responses are not checked for tool_calls.
            # Greetings are bot-initiated and strategies are not expected
            # to request tool calls during greet.  If this assumption
            # changes, add the tool execution loop here (matching
            # chat/stream_chat).
            turn.populate_from_response(response, self.llm)
            await self._finalize_turn(turn)
            return turn.response_content
        except Exception as e:
            await self._call_on_error_middleware(e, "", context)
            raise

    async def stream_chat(
        self,
        message: str,
        context: BotContext,
        temperature: float | None = None,
        max_tokens: int | None = None,
        rag_query: str | None = None,
        llm_config_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMStreamResponse, None]:
        """Stream chat response token by token.

        Similar to chat() but yields ``LLMStreamResponse`` objects as they are
        generated, providing both the text delta and rich metadata (usage,
        finish_reason, is_final) for each chunk.

        Args:
            message: User message to process
            context: Bot execution context
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            rag_query: Optional explicit query for knowledge base retrieval.
                      If provided, this is used instead of the message for RAG.
            llm_config_overrides: Optional dict to override LLM config fields
                      for this request only. Supported fields: model, temperature,
                      max_tokens, top_p, stop_sequences, seed, options.
            **kwargs: Additional arguments passed to LLM

        Yields:
            LLMStreamResponse objects with ``.delta`` (text), ``.is_final``,
            ``.usage``, and ``.finish_reason`` attributes.

        Example:
            ```python
            context = BotContext(
                conversation_id="conv-123",
                client_id="client-456",
                user_id="user-789"
            )

            # Stream and display in real-time
            async for chunk in bot.stream_chat("Explain quantum computing", context):
                print(chunk.delta, end="", flush=True)
            print()  # Newline after streaming

            # Accumulate response
            full_response = ""
            async for chunk in bot.stream_chat("Hello!", context):
                full_response += chunk.delta

            # With LLM config overrides
            async for chunk in bot.stream_chat(
                "Explain quantum computing",
                context,
                llm_config_overrides={"model": "gpt-4-turbo"}
            ):
                print(chunk.delta, end="", flush=True)
            ```

        Note:
            Conversation history is automatically updated after streaming completes.
            When a reasoning_strategy is configured, the strategy produces the
            complete response and it is emitted as a single stream chunk.
        """
        turn = TurnState(
            mode=TurnMode.STREAM,
            message=message,
            context=context,
            rag_query=rag_query,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_config_overrides=llm_config_overrides,
        )
        streaming_error: Exception | None = None

        try:
            await self._prepare_turn(turn)

            # Track tool_calls across streaming rounds so the tool
            # execution loop can pick them up after the initial stream.
            pending_tool_calls: list[Any] | None = None

            if self.reasoning_strategy:
                # Delegate to the strategy's stream_generate().
                # Strategies with true streaming (SimpleReasoning) yield
                # LLMStreamResponse chunks; others yield a single complete
                # response that we wrap as a stream chunk.
                async for chunk in self.reasoning_strategy.stream_generate(
                    manager=turn.manager,
                    llm=self.llm,
                    tools=list(self.tool_registry) or None,
                    temperature=temperature or self.default_temperature,
                    max_tokens=max_tokens or self.default_max_tokens,
                    llm_config_overrides=llm_config_overrides,
                ):
                    if isinstance(chunk, LLMStreamResponse):
                        turn.stream_chunks.append(chunk.delta)
                        if chunk.is_final or chunk.usage:
                            turn.populate_from_final_stream_chunk(
                                chunk, self.llm
                            )
                        # Intercept tool_calls: suppress is_final so the
                        # consumer knows more content may follow.
                        if chunk.tool_calls and self.tool_registry:
                            pending_tool_calls = chunk.tool_calls
                            yield LLMStreamResponse(
                                delta=chunk.delta,
                                is_final=False,
                                usage=chunk.usage,
                                model=chunk.model,
                            )
                        else:
                            yield chunk
                    else:
                        # Strategy yielded a complete LLMResponse — wrap it
                        content = self._extract_response_content(chunk)
                        turn.stream_chunks.append(content)
                        turn.populate_from_response(chunk, self.llm)
                        # Check for tool_calls on the LLMResponse
                        if (
                            getattr(chunk, "tool_calls", None)
                            and self.tool_registry
                        ):
                            pending_tool_calls = chunk.tool_calls
                            yield LLMStreamResponse(
                                delta=content, is_final=False,
                            )
                        else:
                            yield LLMStreamResponse(
                                delta=content,
                                is_final=True,
                                finish_reason="stop",
                            )
            else:
                # No reasoning strategy — stream directly from LLM
                async for chunk in turn.manager.stream_complete(
                    tools=list(self.tool_registry) or None,
                    llm_config_overrides=llm_config_overrides,
                    temperature=temperature or self.default_temperature,
                    max_tokens=max_tokens or self.default_max_tokens,
                    **kwargs,
                ):
                    turn.stream_chunks.append(chunk.delta)
                    if chunk.is_final or chunk.usage:
                        turn.populate_from_final_stream_chunk(chunk, self.llm)
                    if chunk.tool_calls and self.tool_registry:
                        pending_tool_calls = chunk.tool_calls
                        yield LLMStreamResponse(
                            delta=chunk.delta,
                            is_final=False,
                            usage=chunk.usage,
                            model=chunk.model,
                        )
                    else:
                        yield chunk

            # DynaBot-level tool execution loop for streaming.
            # Execute pending tool_calls, then re-stream until no
            # more tool_calls or max iterations reached.
            for _iteration in range(self._max_tool_iterations):
                if not pending_tool_calls or not self.tool_registry:
                    break
                await self._execute_tools(turn, pending_tool_calls)
                # Accumulate usage from intermediate streaming rounds
                turn.accumulate_usage_from_stream()
                pending_tool_calls = None

                async for chunk in turn.manager.stream_complete(
                    tools=list(self.tool_registry) or None,
                    temperature=temperature or self.default_temperature,
                    max_tokens=max_tokens or self.default_max_tokens,
                    llm_config_overrides=llm_config_overrides,
                ):
                    turn.stream_chunks.append(chunk.delta)
                    if chunk.is_final or chunk.usage:
                        turn.populate_from_final_stream_chunk(
                            chunk, self.llm
                        )
                    if chunk.tool_calls and self.tool_registry:
                        pending_tool_calls = chunk.tool_calls
                        yield LLMStreamResponse(
                            delta=chunk.delta,
                            is_final=False,
                            usage=chunk.usage,
                            model=chunk.model,
                        )
                    else:
                        yield chunk
            else:
                # Loop completed without break — cap hit
                if pending_tool_calls and self.tool_registry:
                    logger.warning(
                        "Streaming tool execution loop reached max "
                        "iterations (%d) with pending tool_calls",
                        self._max_tool_iterations,
                        extra={
                            "conversation_id": getattr(
                                turn.manager, "conversation_id", None
                            ),
                        },
                    )

        except Exception as e:
            streaming_error = e
            await self._call_on_error_middleware(e, message, context)
            raise

        # Only finalize on success
        if streaming_error is None:
            turn.response_content = "".join(turn.stream_chunks)
            await self._finalize_turn(turn)

    async def get_conversation(self, conversation_id: str) -> Any:
        """Retrieve conversation history.

        This method fetches the complete conversation state including all messages,
        metadata, and the message tree structure. Useful for displaying conversation
        history, debugging, analytics, or exporting conversations.

        Args:
            conversation_id: Unique identifier of the conversation to retrieve

        Returns:
            ConversationState object containing the full conversation history,
            or None if the conversation does not exist

        Example:
            ```python
            # Retrieve a conversation
            conv_state = await bot.get_conversation("conv-123")

            # Access messages
            messages = conv_state.message_tree

            # Access metadata
            print(conv_state.metadata)
            ```

        See Also:
            - clear_conversation(): Clear/delete a conversation
            - chat(): Add messages to a conversation
        """
        return await self.conversation_storage.load_conversation(conversation_id)

    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation's history.

        This method removes the conversation from both persistent storage and the
        internal cache. The next chat() call with this conversation_id will start
        a fresh conversation. Useful for:

        - Implementing "start over" functionality
        - Privacy/data deletion requirements
        - Testing and cleanup
        - Resetting conversation context

        Args:
            conversation_id: Unique identifier of the conversation to clear

        Returns:
            True if the conversation was deleted, False if it didn't exist

        Example:
            ```python
            # Clear a conversation
            deleted = await bot.clear_conversation("conv-123")

            if deleted:
                print("Conversation deleted")
            else:
                print("Conversation not found")

            # Next chat will start fresh
            response = await bot.chat("Hello!", context)
            ```

        Note:
            This operation is permanent and cannot be undone. The conversation
            cannot be recovered after deletion.

        See Also:
            - get_conversation(): Retrieve conversation before clearing
            - chat(): Will create new conversation after clearing
        """
        # Remove from cache if present
        if conversation_id in self._conversation_managers:
            del self._conversation_managers[conversation_id]

        # Delete from storage
        return await self.conversation_storage.delete_conversation(conversation_id)

    async def get_wizard_state(self, conversation_id: str) -> dict[str, Any] | None:
        """Get current wizard state for a conversation.

        This method provides public access to wizard state without requiring
        access to private conversation managers. It checks the in-memory
        manager first (most current) and falls back to persisted storage.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Wizard state dict with canonical structure, or None if no wizard
            active or conversation not found.

        The returned dict follows the canonical schema:
            {
                "current_stage": str,
                "stage_index": int,
                "total_stages": int,
                "progress": float,
                "completed": bool,
                "data": dict,
                "can_skip": bool,
                "can_go_back": bool,
                "suggestions": list[str],
                "history": list[str],
            }

        Example:
            ```python
            # Get wizard state for a conversation
            state = await bot.get_wizard_state("conv-123")

            if state:
                print(f"Current stage: {state['current_stage']}")
                print(f"Progress: {state['progress'] * 100:.0f}%")
                print(f"Collected data: {state['data']}")
            ```
        """
        # Fast path: in-memory cache
        manager = self._conversation_managers.get(conversation_id)
        if manager and manager.metadata:
            wizard_meta = manager.metadata.get("wizard")
            if wizard_meta:
                return self._normalize_wizard_state(wizard_meta)

        # Slow path: fall back to persisted storage
        state = await self.conversation_storage.load_conversation(conversation_id)
        if state and state.metadata:
            wizard_meta = state.metadata.get("wizard")
            if wizard_meta:
                return self._normalize_wizard_state(wizard_meta)

        return None

    def _normalize_wizard_state(
        self, wizard_meta: dict[str, Any]
    ) -> dict[str, Any]:
        """Normalize wizard metadata to canonical structure.

        Delegates to the module-level ``normalize_wizard_state()`` function.
        """
        return normalize_wizard_state(wizard_meta)

    async def close(self) -> None:
        """Close the bot and clean up resources.

        This method closes the LLM provider, conversation storage backend,
        reasoning strategy, and releases associated resources like HTTP
        connections and database connections. Should be called when the bot
        is no longer needed, especially in testing or when creating temporary
        bot instances.

        Example:
            ```python
            bot = await DynaBot.from_config(config)
            try:
                response = await bot.chat("Hello", context)
            finally:
                await bot.close()
            ```

        Note:
            After calling close(), the bot should not be used for further operations.
            Create a new bot instance if needed.
        """
        # Each subsystem owns the lifecycle of the providers it created.
        # The provider registry is a catalog for observability — it does
        # not manage lifecycle.  DynaBot only closes self.llm (the main
        # provider it created).

        # Close subsystems — each closes its own providers and resources.
        if self.knowledge_base:
            try:
                await self.knowledge_base.close()
            except Exception:
                logger.exception("Error closing knowledge base")

        if self.reasoning_strategy:
            try:
                await self.reasoning_strategy.close()
            except Exception:
                logger.exception("Error closing reasoning strategy")

        if self.memory:
            try:
                await self.memory.close()
            except Exception:
                logger.exception("Error closing memory store")

        # Close conversation storage
        if self.conversation_storage:
            try:
                await self.conversation_storage.close()
            except Exception:
                logger.exception("Error closing conversation storage")

        # Close main LLM provider only if DynaBot created it.
        # When from_config(llm=...) was used, the caller owns the lifecycle.
        if self._owns_llm and self.llm and hasattr(self.llm, "close"):
            try:
                await self.llm.close()
            except Exception:
                logger.exception("Error closing main LLM provider")

    async def __aenter__(self) -> Self:
        """Async context manager entry.

        Returns:
            Self for use in async with statement
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit - ensures cleanup.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        await self.close()

    def get_conversation_manager(
        self, conversation_id: str
    ) -> ConversationManager | None:
        """Get a cached conversation manager by conversation ID.

        Returns ``None`` if no manager exists for the given ID (i.e. no
        turn has been processed for that conversation yet).  Use this for
        cross-layer integration testing (e.g. injecting LLM-layer
        ``ConversationMiddleware`` into a manager after construction).

        Args:
            conversation_id: Conversation identifier

        Returns:
            Cached ConversationManager, or None
        """
        return self._conversation_managers.get(conversation_id)

    async def _get_or_create_conversation(
        self, context: BotContext
    ) -> ConversationManager:
        """Get or create conversation manager for context.

        Args:
            context: Bot execution context

        Returns:
            ConversationManager instance
        """
        conv_id = context.conversation_id

        # Check cache
        if conv_id in self._conversation_managers:
            return self._conversation_managers[conv_id]

        # Try to resume existing conversation
        try:
            manager = await ConversationManager.resume(
                conversation_id=conv_id,
                llm=self.llm,
                prompt_builder=self.prompt_builder,
                storage=self.conversation_storage,
            )
        except Exception:
            metadata = {
                "client_id": context.client_id,
                "user_id": context.user_id,
                "model": self.llm.config.model,
                "provider": self.llm.config.provider,
                "tools": self.tool_registry.get_tool_names(),
                **context.session_metadata,
            }

            manager = ConversationManager(
                llm=self.llm,
                prompt_builder=self.prompt_builder,
                storage=self.conversation_storage,
                conversation_id=conv_id,
                metadata=metadata,
            )

            if self.system_prompt_name:
                await manager.add_message(
                    prompt_name=self.system_prompt_name,
                    role="system",
                )
            elif self.system_prompt_content:
                await manager.add_message(
                    content=self.system_prompt_content,
                    role="system",
                    rag_configs=self.system_prompt_rag_configs,
                    include_rag=bool(self.system_prompt_rag_configs),
                )

        # Cache manager
        self._conversation_managers[conv_id] = manager
        return manager

    async def _build_message_with_context(
        self,
        message: str,
        rag_query: str | None = None,
    ) -> str:
        """Build message with knowledge and memory context.

        Args:
            message: Original user message
            rag_query: Optional explicit query for knowledge base retrieval.
                      If provided, this is used instead of the message for RAG.

        Returns:
            Message augmented with context
        """
        contexts = []

        # Add knowledge context (skip when auto_context is disabled —
        # KB remains available for tool-based access)
        if self.knowledge_base and self._kb_auto_context:
            # Use explicit rag_query if provided, otherwise use message
            search_query = rag_query if rag_query else message
            kb_results = await self.knowledge_base.query(search_query, k=5)
            if kb_results:
                kb_context = self.knowledge_base.format_context(
                    kb_results, wrap_in_tags=True
                )
                if self._context_transform:
                    kb_context = self._context_transform(kb_context)
                contexts.append(kb_context)

        # Add memory context
        if self.memory:
            mem_results = await self.memory.get_context(message)
            if mem_results:
                mem_context = "\n\n".join([r["content"] for r in mem_results])
                if self._context_transform:
                    mem_context = self._context_transform(mem_context)
                contexts.append(f"<conversation_history>\n{mem_context}\n</conversation_history>")

        # Build full message with clear separation
        if contexts:
            context_section = "\n\n".join(contexts)
            return f"{context_section}\n\n<question>\n{message}\n</question>"
        return message

    @staticmethod
    def _resolve_tool(
        tool_config: dict[str, Any] | str,
        config: dict[str, Any],
        dependencies: dict[str, Any] | None = None,
    ) -> Any | None:
        """Resolve tool from configuration.

        Supports two patterns:
        1. Direct class instantiation: {"class": "module.ToolClass", "params": {...}}
        2. XRef resolution: "xref:tools[tool_name]" or {"xref": "tools[tool_name]"}

        For direct instantiation, if the tool class defines a
        ``from_config(cls, config: dict)`` classmethod, it will be
        called with ``params`` instead of ``tool_class(**params)``.
        This allows tools to construct complex internal dependencies
        from simple YAML-compatible parameters.

        If the tool class defines ``catalog_metadata()`` with a ``requires``
        tuple, matching entries from ``dependencies`` are injected into
        the constructor parameters (unless already provided in ``params``).

        Args:
            tool_config: Tool configuration (dict or string xref)
            config: Full bot configuration for xref resolution
            dependencies: Optional resource dependencies to inject into tools
                that declare them via catalog_metadata().requires

        Returns:
            Tool instance or None if resolution fails

        Example:
            # Direct instantiation
            tool_config = {
                "class": "my_tools.CalculatorTool",
                "params": {"precision": 2}
            }

            # XRef to pre-defined tool
            tool_config = "xref:tools[calculator]"
            # Requires config to have:
            # {
            #     "tool_definitions": {
            #         "calculator": {
            #             "class": "my_tools.CalculatorTool",
            #             "params": {}
            #         }
            #     }
            # }
        """
        import importlib
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Handle xref string format
            if isinstance(tool_config, str):
                if tool_config.startswith("xref:"):
                    # Parse xref (e.g., "xref:tools[calculator]")
                    # Extract the reference name
                    import re

                    match = re.match(r"xref:tools\[([^\]]+)\]", tool_config)
                    if not match:
                        logger.error(f"Invalid xref format: {tool_config}")
                        return None

                    tool_name = match.group(1)

                    # Look up in tool_definitions
                    tool_definitions = config.get("tool_definitions", {})
                    if tool_name not in tool_definitions:
                        logger.error(
                            f"Tool definition not found: {tool_name}. "
                            f"Available: {list(tool_definitions.keys())}"
                        )
                        return None

                    # Recursively resolve the referenced config
                    return DynaBot._resolve_tool(tool_definitions[tool_name], config, dependencies)
                else:
                    logger.error(f"String tool config must be xref format: {tool_config}")
                    return None

            # Handle dict with xref key
            if isinstance(tool_config, dict) and "xref" in tool_config:
                return DynaBot._resolve_tool(tool_config["xref"], config, dependencies)

            # Handle dict with class key (direct instantiation)
            if isinstance(tool_config, dict) and "class" in tool_config:
                class_path = tool_config["class"]
                params = tool_config.get("params", {})

                # Import the tool class
                module_path, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                tool_class = getattr(module, class_name)

                # Inject dependencies declared in catalog_metadata().requires
                if dependencies:
                    meta_fn = getattr(tool_class, "catalog_metadata", None)
                    if meta_fn and callable(meta_fn):
                        requires = meta_fn().get("requires") or ()
                        for dep_name in requires:
                            if dep_name in dependencies and dep_name not in params:
                                params[dep_name] = dependencies[dep_name]

                # Instantiate the tool — prefer from_config() if available,
                # which allows tools to construct complex internal
                # dependencies from simple YAML-compatible params.
                if hasattr(tool_class, "from_config") and callable(
                    tool_class.from_config
                ):
                    tool = tool_class.from_config(params)
                else:
                    tool = tool_class(**params)

                # Validate it's a Tool instance
                from dataknobs_llm.tools import Tool

                if not isinstance(tool, Tool):
                    logger.error(
                        f"Resolved class {class_path} is not a Tool instance: {type(tool)}"
                    )
                    return None

                logger.info(f"Successfully loaded tool: {tool.name} ({class_path})")
                return tool
            else:
                logger.error(
                    f"Invalid tool config format. Expected dict with 'class' or 'xref' key, "
                    f"or xref string. Got: {type(tool_config)}"
                )
                return None

        except ImportError as e:
            logger.error(f"Failed to import tool class: {e}")
            return None
        except AttributeError as e:
            logger.error(f"Failed to find tool class: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to instantiate tool: {e}")
            return None

    @staticmethod
    def _create_middleware(config: dict[str, Any]) -> Middleware | None:
        """Create middleware from configuration.

        Args:
            config: Middleware configuration

        Returns:
            Middleware instance or None
        """
        try:
            import importlib

            module_path, class_name = config["class"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            middleware_class = getattr(module, class_name)
            return middleware_class(**config.get("params", {}))
        except Exception:
            return None

    # -----------------------------------------------------------------
    # Undo / Rewind
    # -----------------------------------------------------------------

    async def undo_last_turn(self, context: BotContext) -> UndoResult:
        """Undo the last conversational turn (user message + bot response).

        Navigates the conversation tree back to the node_id recorded before
        the last turn started. The next chat() call will create a new branch
        from that point. The original branch is preserved in the tree.

        Also rolls back:
        - Memory layer (pop N messages based on node depth difference)
        - Wizard FSM state (restored from per-node metadata)
        - Memory banks (reverted via backend-managed checkpointing)

        Args:
            context: Bot execution context (identifies the conversation).

        Returns:
            UndoResult with details about what was undone.

        Raises:
            ValueError: If there's nothing to undo (at start of conversation).
        """
        conv_id = context.conversation_id
        manager = self._conversation_managers.get(conv_id)
        if manager is None or manager.state is None:
            raise ValueError("No active conversation")

        checkpoints = self._turn_checkpoints.get(conv_id, [])
        if not checkpoints:
            raise ValueError("Nothing to undo")

        checkpoint_node_id, checkpoint_mem_count = checkpoints.pop()

        # Identify what we're undoing (last user message + last bot response).
        # For user messages, prefer raw_content from node metadata so that
        # UndoResult.undone_user_message reflects the original user input
        # rather than the KB/memory-augmented version.
        undone_user = ""
        undone_bot = ""
        nodes = manager.state.get_current_nodes()
        for node in reversed(nodes):
            role = node.message.role
            if role == "assistant" and not undone_bot:
                content = node.message.content
                undone_bot = content if isinstance(content, str) else str(content)
            elif role == "user" and not undone_user:
                raw = node.metadata.get("raw_content")
                if raw is not None:
                    undone_user = raw
                else:
                    content = node.message.content
                    undone_user = content if isinstance(content, str) else str(content)
                break

        # Navigate back — next add_message() creates a sibling branch
        await manager.switch_to_node(checkpoint_node_id)

        # Roll back memory — use stored message count for accuracy
        current_mem_count = 0
        if self.memory:
            try:
                current_mem_count = len(await self.memory.get_context(""))
            except Exception:
                current_mem_count = 0
        messages_to_pop = current_mem_count - checkpoint_mem_count
        if self.memory and messages_to_pop > 0:
            try:
                await self.memory.pop_messages(messages_to_pop)
            except (ValueError, NotImplementedError):
                logger.warning(
                    "Memory pop_messages failed for %d messages",
                    messages_to_pop,
                    exc_info=True,
                )

        # Restore wizard FSM state from checkpoint node's metadata
        self._restore_wizard_from_node(manager, checkpoint_node_id)

        # Revert banks via backend-managed checkpointing
        self._undo_banks_to_checkpoint(checkpoint_node_id)

        # Count remaining turns
        remaining_messages = manager.messages
        user_count = sum(
            1 for m in remaining_messages
            if (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "user"
        )

        return UndoResult(
            undone_user_message=undone_user,
            undone_bot_response=undone_bot,
            remaining_turns=user_count,
            branching=True,
        )

    async def rewind_to_turn(
        self, context: BotContext, turn: int
    ) -> UndoResult:
        """Rewind conversation to after the given turn number.

        Turn 0 is the first user-bot exchange. Rewinding to turn -1
        means back to the start (before any user messages).

        Args:
            context: Bot execution context.
            turn: Turn number to rewind to (-1 for conversation start).

        Returns:
            UndoResult with details about what was undone.

        Raises:
            ValueError: If turn number is invalid.
        """
        conv_id = context.conversation_id
        checkpoints = self._turn_checkpoints.get(conv_id, [])
        target_count = turn + 1  # checkpoints[0] is before turn 0

        if target_count < 0 or target_count > len(checkpoints):
            raise ValueError(
                f"Invalid turn {turn}: conversation has "
                f"{len(checkpoints)} turns"
            )

        turns_to_undo = len(checkpoints) - target_count
        result = None
        for _ in range(turns_to_undo):
            result = await self.undo_last_turn(context)

        if result is None:
            raise ValueError("Nothing to undo")
        return result

    def _restore_wizard_from_node(
        self, manager: ConversationManager, node_id: str
    ) -> None:
        """Restore wizard FSM state from a checkpoint node's metadata.

        Reads ``wizard_fsm_state`` from the node's metadata and restores
        the wizard reasoning strategy to that state.

        Args:
            manager: ConversationManager with the conversation tree.
            node_id: Node ID to restore FSM state from.
        """
        if not self.reasoning_strategy:
            return
        if not hasattr(self.reasoning_strategy, "_get_wizard_state"):
            return
        if manager.state is None:
            return

        node = get_node_by_id(manager.state.message_tree, node_id)
        if node is None:
            return

        node_data = node.data
        if not isinstance(node_data, ConversationNode):
            return

        fsm_state = node_data.metadata.get("wizard_fsm_state")
        if not fsm_state:
            return

        # Write the FSM state to conversation-level metadata so
        # _get_wizard_state() can pick it up on the next generate() call.
        wizard_meta = manager.metadata.get("wizard", {})
        wizard_meta["fsm_state"] = fsm_state

        # Also update top-level keys that normalize_wizard_state() reads
        # with higher priority than fsm_state.  Without this, stale values
        # from the pre-undo turn would shadow the restored fsm_state.
        wizard_meta["current_stage"] = fsm_state.get("current_stage")
        wizard_meta["data"] = fsm_state.get("data", {})
        wizard_meta["completed"] = fsm_state.get("completed", False)
        wizard_meta["history"] = fsm_state.get("history", [])

        manager.metadata["wizard"] = wizard_meta

    def _undo_banks_to_checkpoint(self, checkpoint_node_id: str) -> None:
        """Revert memory banks to the checkpoint via undo_to_checkpoint().

        Args:
            checkpoint_node_id: Node ID to revert banks to.
        """
        if not self.reasoning_strategy:
            return
        banks = getattr(self.reasoning_strategy, "_banks", None)
        if not banks:
            return
        for bank in banks.values():
            if hasattr(bank, "undo_to_checkpoint"):
                bank.undo_to_checkpoint(checkpoint_node_id)
