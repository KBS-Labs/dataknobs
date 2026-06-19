"""Core DynaBot implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, TypeVar

from dataknobs_common.exceptions import ConfigurationError, NotFoundError
from dataknobs_common.lifecycle import close_if_owned
from dataknobs_common.structured_config import StructuredConfigConsumer
from dataknobs_llm import LLMStreamResponse
from dataknobs_llm.conversations import (
    ConversationManager,
    ConversationMiddleware,
    ConversationStorage,
    DataknobsConversationStorage,
)
from dataknobs_llm.conversations.storage import ConversationNode, get_node_by_id
from dataknobs_llm.llm import AsyncLLMProvider
from dataknobs_llm.prompts import AsyncPromptBuilder
from dataknobs_llm.tools import ToolRegistry
from dataknobs_llm.tools.context import ToolExecutionContext

from ..knowledge.base import KnowledgeBase
from ..memory.base import Memory
from ..middleware.base import Middleware
from .config import DynaBotConfig
from .context import BotContext
from .turn import ToolExecution, TurnMode, TurnState

if TYPE_CHECKING:
    from dataknobs_config import EnvironmentAwareConfig, EnvironmentConfig

    from ..prompts.resolver import PromptResolver
    from ..reasoning.base import ProcessResult

logger = logging.getLogger(__name__)

# Provider role constants for the provider registry.
PROVIDER_ROLE_MAIN = "main"
PROVIDER_ROLE_EXTRACTION = "extraction"
PROVIDER_ROLE_MEMORY_EMBEDDING = "memory_embedding"
PROVIDER_ROLE_SUMMARY_LLM = "summary_llm"
PROVIDER_ROLE_KB_EMBEDDING = "kb_embedding"

# Type variable for ``DynaBot.get_steps_of_type`` — preserves the caller's
# requested class as the element type of the returned list.
_StepT = TypeVar("_StepT")


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


class DynaBot(StructuredConfigConsumer[DynaBotConfig]):
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

    CONFIG_CLS = DynaBotConfig

    _DEFAULT_MAX_TOOL_ITERATIONS = 5
    """Default maximum number of tool execution rounds before returning."""

    _DEFAULT_TOOL_TIMEOUT: float = 30.0
    """Default per-tool execution timeout in seconds."""

    _DEFAULT_TOOL_LOOP_TIMEOUT: float = 120.0
    """Default wall-clock timeout for the entire tool execution loop."""

    def __init__(
        self,
        llm: AsyncLLMProvider | DynaBotConfig | Mapping[str, Any] | None = None,
        prompt_builder: AsyncPromptBuilder | None = None,
        conversation_storage: ConversationStorage | None = None,
        tool_registry: ToolRegistry | None = None,
        memory: Memory | None = None,
        knowledge_base: KnowledgeBase | None = None,
        kb_auto_context: bool = True,
        reasoning_strategy: Any | None = None,
        middleware: list[Middleware] | None = None,
        conversation_middleware: list[ConversationMiddleware] | None = None,
        system_prompt_name: str | None = None,
        system_prompt_content: str | None = None,
        system_prompt_rag_configs: list[dict[str, Any]] | None = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 1000,
        context_transform: Callable[[str], str] | None = None,
        max_tool_iterations: int = _DEFAULT_MAX_TOOL_ITERATIONS,
        tool_timeout: float = _DEFAULT_TOOL_TIMEOUT,
        tool_loop_timeout: float = _DEFAULT_TOOL_LOOP_TIMEOUT,
        prompt_resolver: PromptResolver | None = None,
        prompt_envelope: str | None = None,
        *,
        _components: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize DynaBot.

        This is a **dual-input** constructor with two distinct shapes:

        - **Pre-built collaborators** (``llm`` is an
          :class:`~dataknobs_llm.llm.AsyncLLMProvider` instance):
          ``DynaBot(llm=provider, prompt_builder=..., conversation_storage=...)``
          assigns the already-built collaborators directly. This is the
          public programmatic constructor used by tests, ``BotTestHarness``,
          and advanced callers; :meth:`from_components` is its canonical
          named alias.
        - **Config-driven** (``llm`` is a ``DynaBotConfig`` or ``Mapping``):
          the typed-config construction lifecycle established by
          ``StructuredConfigConsumer`` runs — ``self._config`` is set and
          :meth:`_setup` is called, with the collaborators built later by
          :meth:`_ainit`. This path is normally reached through
          :meth:`from_config` / ``from_config_async``, not called directly.

        Args:
            llm: A pre-built LLM provider (pre-built shape) or a
                ``DynaBotConfig`` / config mapping (config-driven shape).
            prompt_builder: Prompt builder instance (pre-built shape).
            conversation_storage: Conversation storage backend (pre-built shape).
            tool_registry: Optional tool registry
            memory: Optional memory implementation
            knowledge_base: Optional knowledge base
            kb_auto_context: Whether to auto-inject KB results into messages.
                When False, the KB is still available for tool-based access
                but not automatically queried on every message.
            reasoning_strategy: Optional reasoning strategy
            middleware: Optional list of bot-turn ``Middleware`` instances
                (``dataknobs_bots.middleware.Middleware`` — turn-lifecycle
                hooks).
            conversation_middleware: Optional list of LLM-call
                ``ConversationMiddleware`` instances
                (``dataknobs_llm.conversations.ConversationMiddleware`` —
                request/response wraps around ``llm.complete``). Forwarded
                to every :class:`~dataknobs_llm.conversations.ConversationManager`
                this bot constructs. Distinct from ``middleware`` because
                the two interfaces are structurally different.
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
            tool_timeout: Per-tool execution timeout in seconds.  If a
                single tool call exceeds this duration, it is cancelled
                and an error observation is recorded.
            tool_loop_timeout: Wall-clock budget in seconds for the
                tool execution loop (across all iterations).  Checked
                at the start of each iteration and before each LLM
                re-call.  For ``chat()``, the LLM re-call is also
                bounded by the remaining budget via
                ``asyncio.wait_for()``.  For ``stream_chat()``, a
                streaming re-call that starts within budget runs to
                completion (async generators cannot be reliably
                cancelled mid-chunk).  Individual tool executions are always
                bounded by ``tool_timeout``.
            prompt_resolver: Optional :class:`PromptResolver` for resolving
                prompts from the composed prompt library.  Built automatically
                by ``from_config()``; pass explicitly only when constructing
                bots programmatically with custom libraries.
            prompt_envelope: Envelope style applied to the auto-context
                user prompt and the grounded-reasoning synthesis-prompt KB
                block. One of ``"markdown"`` (default), ``"xml"``, or
                ``"prose"`` (case-insensitive). Mirrors
                ``DynaBotConfig.prompt_envelope`` for the pre-built shape
                so a programmatically-constructed bot can pin the legacy
                ``"xml"`` shape without going through a config mapping.
        """
        # Config-driven shape: from_config / from_config_async / from_components
        # deliver a typed DynaBotConfig (or a config Mapping) in the first
        # positional slot. Route it through the structured-config mixin, which
        # establishes self._config and runs _setup; collaborators are built
        # later by _ainit (or adopted by _adopt_components).
        if isinstance(llm, (DynaBotConfig, Mapping)):
            stray = {
                name: value
                for name, value in (
                    ("prompt_builder", prompt_builder),
                    ("conversation_storage", conversation_storage),
                    ("tool_registry", tool_registry),
                    ("memory", memory),
                    ("knowledge_base", knowledge_base),
                    ("reasoning_strategy", reasoning_strategy),
                    ("middleware", middleware),
                    ("conversation_middleware", conversation_middleware),
                    ("system_prompt_name", system_prompt_name),
                    ("system_prompt_content", system_prompt_content),
                    ("system_prompt_rag_configs", system_prompt_rag_configs),
                    ("context_transform", context_transform),
                    ("prompt_resolver", prompt_resolver),
                    ("prompt_envelope", prompt_envelope),
                )
                if value is not None
            }
            if stray:
                raise TypeError(
                    f"{type(self).__name__}: cannot mix a config "
                    f"({type(llm).__name__}) with pre-built collaborator "
                    f"arguments {sorted(stray)}. Pass the config alone "
                    "(collaborators are built from it) or use the pre-built "
                    "constructor form with an AsyncLLMProvider."
                )
            super().__init__(llm, _components=_components, **kwargs)
            return

        # Pre-built shape: an AsyncLLMProvider (or None) in the first slot.
        if llm is None:
            raise TypeError(
                f"{type(self).__name__}: `llm` is required — pass an "
                "AsyncLLMProvider for direct construction, or a DynaBotConfig "
                "/ config mapping for config-driven construction."
            )
        # Build a typed snapshot of the scalar knobs for self.config. The
        # effective default temperature / max tokens live in the llm section
        # (mirroring the config-driven path), so _setup derives them from
        # there for both shapes. A callable context_transform is not
        # serializable, so it is omitted from the snapshot and assigned to
        # the live attribute below.
        # Build snapshot kwargs, omitting prompt_envelope when the caller
        # did not specify it so the DynaBotConfig default applies (one
        # source of truth for the default; pre-built and config paths
        # cannot drift).
        snapshot_kwargs: dict[str, Any] = {
            "llm": {
                "temperature": default_temperature,
                "max_tokens": default_max_tokens,
            },
            "max_tool_iterations": max_tool_iterations,
            "tool_timeout": tool_timeout,
            "tool_loop_timeout": tool_loop_timeout,
            "context_transform": (
                context_transform if isinstance(context_transform, str) else None
            ),
        }
        if prompt_envelope is not None:
            snapshot_kwargs["prompt_envelope"] = prompt_envelope
        snapshot = DynaBotConfig(**snapshot_kwargs)
        super().__init__(snapshot, _components=_components)
        self._prebuilt = True
        self._assign_collaborators(
            llm=llm,
            prompt_builder=prompt_builder,
            conversation_storage=conversation_storage,
            tool_registry=tool_registry,
            memory=memory,
            knowledge_base=knowledge_base,
            kb_auto_context=kb_auto_context,
            reasoning_strategy=reasoning_strategy,
            middleware=middleware,
            conversation_middleware=conversation_middleware,
            system_prompt_name=system_prompt_name,
            system_prompt_content=system_prompt_content,
            system_prompt_rag_configs=system_prompt_rag_configs,
            context_transform=context_transform,
            prompt_resolver=prompt_resolver,
            owns_llm=True,
        )

    def _setup(self) -> None:
        """Derive cheap, sync state from ``self.config`` (both shapes).

        Runs during ``__init__`` for both the pre-built and config-driven
        shapes. Establishes only derived scalars and empty per-conversation
        caches — no provider/KB/memory construction. Those collaborators are
        built by the async :meth:`_ainit` body (config-driven shape) or
        assigned by :meth:`_assign_collaborators` (pre-built shape), both of
        which run after ``__init__``; nothing reads them in between.
        """
        llm_config = self.config.llm or {}
        self.default_temperature = llm_config.get("temperature", 0.7)
        self.default_max_tokens = llm_config.get("max_tokens", 1000)
        self._max_tool_iterations = self.config.max_tool_iterations
        self._tool_timeout = self.config.tool_timeout
        self._tool_loop_timeout = self.config.tool_loop_timeout
        # Resolve the dotted-path context_transform now (cheap, sync). The
        # pre-built shape overrides this with a directly-supplied callable.
        self._context_transform = self._resolve_context_transform(
            self.config.context_transform
        )
        # Build the prompt envelope once. The string value was validated
        # by DynaBotConfig.__post_init__, so the enum lookup here cannot
        # raise on a valid snapshot.
        from dataknobs_bots.prompts.envelope import (
            PromptEnvelope,
            PromptEnvelopeStyle,
        )

        self._prompt_envelope = PromptEnvelope(
            PromptEnvelopeStyle(self.config.prompt_envelope)
        )
        self._conversation_managers: dict[str, ConversationManager] = {}
        self._turn_checkpoints: dict[str, list[tuple[str, int]]] = {}
        self._providers: dict[str, AsyncLLMProvider] = {}
        # Lifetime ownership of the cascade-closed subsystems. Set True by
        # :meth:`_build_collaborators` (config-driven build → the bot
        # created them) and left False on the pre-built / from_components
        # path (collaborators handed in by the caller → caller-owned).
        # ``close()`` only tears down subsystems this bot owns, so a KB /
        # storage / memory / strategy shared across several bots survives
        # one bot's close. (The main ``llm`` keeps its own ``_owns_llm``
        # gate, set by the construction path.)
        self._owns_knowledge_base = False
        self._owns_memory = False
        self._owns_reasoning_strategy = False
        self._owns_conversation_storage = False

    def _assign_collaborators(
        self,
        *,
        llm: AsyncLLMProvider,
        prompt_builder: AsyncPromptBuilder | None = None,
        conversation_storage: ConversationStorage | None = None,
        tool_registry: ToolRegistry | None = None,
        memory: Memory | None = None,
        knowledge_base: KnowledgeBase | None = None,
        kb_auto_context: bool = True,
        reasoning_strategy: Any | None = None,
        middleware: list[Middleware] | None = None,
        conversation_middleware: list[ConversationMiddleware] | None = None,
        system_prompt_name: str | None = None,
        system_prompt_content: str | None = None,
        system_prompt_rag_configs: list[dict[str, Any]] | None = None,
        context_transform: Callable[[str], str] | str | None = None,
        prompt_resolver: PromptResolver | None = None,
        owns_llm: bool = True,
    ) -> None:
        """Bind fully-built collaborators onto ``self``.

        Shared by the pre-built ``__init__`` shape and
        :meth:`_adopt_components` so the two pre-built entry points cannot
        drift. Scalar derived state is set by :meth:`_setup`; this method
        only assigns the collaborator objects.

        ``prompt_builder`` and ``conversation_storage`` are required: a
        functioning bot drives every conversation through a
        :class:`~dataknobs_llm.conversations.ConversationManager`, which
        needs both. Pre-built construction that omits either yields a bot
        that would fail on the first :meth:`chat`, so it is rejected up
        front (mirroring the ``llm`` requirement). The config-driven shape
        builds these in :meth:`_build_collaborators` instead and never
        reaches this method.
        """
        missing = [
            name
            for name, value in (
                ("prompt_builder", prompt_builder),
                ("conversation_storage", conversation_storage),
            )
            if value is None
        ]
        if missing:
            raise TypeError(
                f"{type(self).__name__}: pre-built construction requires "
                f"{' and '.join(missing)} — a built bot needs a prompt "
                "builder and conversation storage. Provide them, or use "
                "config-driven construction via from_config()."
            )
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.conversation_storage = conversation_storage
        self.tool_registry = tool_registry or ToolRegistry()
        self.memory = memory
        self.knowledge_base = knowledge_base
        self._kb_auto_context = kb_auto_context
        self.reasoning_strategy = reasoning_strategy
        self.middleware = middleware or []
        # LLM-call middleware list (ConversationMiddleware), distinct from
        # bot-turn `self.middleware`. Forwarded to every ConversationManager
        # this bot constructs. Set on BOTH the pre-built and config-driven
        # paths so `_create_conversation_manager` can read it unconditionally.
        self._conversation_middleware = conversation_middleware or []
        self.system_prompt_name = system_prompt_name
        self.system_prompt_content = system_prompt_content
        self.system_prompt_rag_configs = system_prompt_rag_configs
        if context_transform is not None:
            # A directly-supplied callable wins over the dotted-path form
            # already resolved by _setup; a string is resolved here.
            self._context_transform = (
                self._resolve_context_transform(context_transform)
                if isinstance(context_transform, str)
                else context_transform
            )
        self._prompt_resolver = prompt_resolver
        self._owns_llm = owns_llm

    def _adopt_components(
        self,
        *,
        llm: AsyncLLMProvider | None = None,
        prompt_builder: AsyncPromptBuilder | None = None,
        conversation_storage: ConversationStorage | None = None,
        tool_registry: ToolRegistry | None = None,
        memory: Memory | None = None,
        knowledge_base: KnowledgeBase | None = None,
        kb_auto_context: bool = True,
        reasoning_strategy: Any | None = None,
        middleware: list[Middleware] | None = None,
        conversation_middleware: list[ConversationMiddleware] | None = None,
        system_prompt_name: str | None = None,
        system_prompt_content: str | None = None,
        system_prompt_rag_configs: list[dict[str, Any]] | None = None,
        context_transform: Callable[[str], str] | str | None = None,
        prompt_resolver: PromptResolver | None = None,
        **_: Any,
    ) -> None:
        """Adopt pre-built collaborators injected via :meth:`from_components`.

        The named alias of the pre-built ``__init__`` shape — delegates to
        :meth:`_assign_collaborators` so both share one assignment path.
        """
        if llm is None:
            raise TypeError(
                f"{type(self).__name__}.from_components requires a built `llm` "
                "collaborator (an AsyncLLMProvider)."
            )
        self._assign_collaborators(
            llm=llm,
            prompt_builder=prompt_builder,
            conversation_storage=conversation_storage,
            tool_registry=tool_registry,
            memory=memory,
            knowledge_base=knowledge_base,
            kb_auto_context=kb_auto_context,
            reasoning_strategy=reasoning_strategy,
            middleware=middleware,
            conversation_middleware=conversation_middleware,
            system_prompt_name=system_prompt_name,
            system_prompt_content=system_prompt_content,
            system_prompt_rag_configs=system_prompt_rag_configs,
            context_transform=context_transform,
            prompt_resolver=prompt_resolver,
            owns_llm=True,
        )

    @staticmethod
    def _resolve_context_transform(
        ref: Callable[[str], str] | str | None,
    ) -> Callable[[str], str] | None:
        """Resolve a context_transform reference to a callable (or ``None``).

        A callable passes through; a dotted import string is resolved; any
        other value is ignored with a warning.
        """
        if ref is None:
            return None
        if callable(ref):
            return ref
        if isinstance(ref, str):
            from dataknobs_bots.tools.resolve import resolve_callable

            try:
                return resolve_callable(ref)
            except (ImportError, AttributeError, ValueError) as exc:
                raise ConfigurationError(
                    f"context_transform: could not resolve '{ref}': {exc}"
                ) from exc
        logger.warning(
            "context_transform must be a callable or dotted import "
            "string, got %s — ignoring",
            type(ref).__name__,
        )
        return None

    @property
    def prompt_resolver(self) -> PromptResolver | None:
        """The prompt resolver for this bot, if configured."""
        return self._prompt_resolver

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
    async def from_config(  # type: ignore[override]
        cls,
        config: Mapping[str, Any] | DynaBotConfig,
        *,
        llm: AsyncLLMProvider | None = None,
        middleware: list[Middleware] | None = None,
        conversation_middleware: list[ConversationMiddleware] | None = None,
        reasoning_components: Mapping[str, Any] | None = None,
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
                - conversation_middleware: Optional ConversationMiddleware
                  configurations (ignored when the
                  ``conversation_middleware`` kwarg is provided)
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
            middleware: Pre-built bot-turn middleware list
                (``dataknobs_bots.middleware.Middleware``).  When provided,
                replaces any ``middleware`` defined in config.
            conversation_middleware: Pre-built LLM-call middleware list
                (``dataknobs_llm.conversations.ConversationMiddleware``).
                When provided, replaces any ``conversation_middleware``
                defined in config.  Use this to inject a shared / per-test
                middleware instance against an otherwise config-driven bot.
            reasoning_components: Optional mapping of consumer-supplied
                collaborators forwarded into the reasoning strategy's
                ``StructuredConfigConsumer.components`` channel at
                construction time. Strategies pick up the keys they read
                (e.g. ``ReActReasoning`` reads ``extra_context`` /
                ``artifact_registry`` / ``review_executor`` /
                ``context_builder`` / ``prompt_refresher``); unknown keys
                are silently absorbed onto ``strategy.components`` and
                ignored. Bot-managed component names
                (``knowledge_base``, ``prompt_resolver``,
                ``prompt_envelope``) raise ``ConfigurationError`` on
                collision — supply them via the corresponding config
                sections.

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

            # With pre-built conversation_middleware (LLM-call wraps)
            bot = await DynaBot.from_config(
                config,
                conversation_middleware=[HistoryRedactionMiddleware(...)],
            )
            ```
        """
        components: dict[str, Any] = {}
        if llm is not None:
            components["llm"] = llm
        if middleware is not None:
            components["middleware"] = middleware
        if conversation_middleware is not None:
            components["conversation_middleware"] = conversation_middleware
        if reasoning_components is not None:
            components["reasoning_components"] = reasoning_components
        # Async-canonical construction: route through the structured-config
        # lifecycle (from_config_async → __init__ → _setup → _ainit) rather
        # than returning a half-built instance. The `llm` / `middleware` /
        # `conversation_middleware` / `reasoning_components` kwargs travel
        # the injected-collaborator channel to _ainit.
        return await cls.from_config_async(config, **components)

    async def _ainit(
        self,
        *,
        llm: AsyncLLMProvider | None = None,
        middleware: list[Middleware] | None = None,
        conversation_middleware: list[ConversationMiddleware] | None = None,
        reasoning_components: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> None:
        """Async build: create/adopt the provider, then build collaborators.

        The body of construction for the config-driven shape. When ``llm`` is
        injected the caller owns its lifecycle; otherwise a provider is
        created from ``self.config.llm`` and closed if any later build step
        raises (so a failed build never leaks an initialized provider).
        Short-circuits for the pre-built shape (collaborators already wired).
        """
        if self._prebuilt:
            return

        if llm is not None:
            # Caller-owned provider — skip creation/initialization.
            self.llm = llm
            self._owns_llm = False
            await self._build_collaborators(
                middleware_override=middleware,
                conversation_middleware_override=conversation_middleware,
                reasoning_components=reasoning_components,
            )
            return

        if not self.config.llm:
            raise ConfigurationError(
                "DynaBot config-driven construction requires an 'llm' "
                "section (at minimum 'provider' and 'model'), or a pre-built "
                "provider passed as from_config(config, llm=...). Neither was "
                "provided."
            )

        from dataknobs_llm.llm import LLMProviderFactory

        created_llm = LLMProviderFactory(is_async=True).create(self.config.llm)
        await created_llm.initialize()
        self.llm = created_llm
        self._owns_llm = True

        # Everything below can fail; ensure the provider is closed on error
        # so we don't leak aiohttp sessions or other resources.
        try:
            await self._build_collaborators(
                middleware_override=middleware,
                conversation_middleware_override=conversation_middleware,
                reasoning_components=reasoning_components,
            )
        except Exception:
            await created_llm.close()
            raise

    async def _build_collaborators(
        self,
        *,
        middleware_override: list[Middleware] | None = None,
        conversation_middleware_override: list[ConversationMiddleware] | None = None,
        reasoning_components: Mapping[str, Any] | None = None,
    ) -> None:
        """Build and bind the configured collaborators onto ``self``.

        Runs after the main LLM provider is set (by :meth:`_ainit`).
        Populates conversation storage, the composed prompt library and
        resolver, memory, knowledge base, tools, reasoning strategy,
        middleware, and the system-prompt fields from ``self.config``.
        Separated so :meth:`_ainit` can guarantee cleanup of an
        internally-created provider if anything here raises.

        Consumer-supplied ``reasoning_components`` are forwarded into the
        strategy's components channel (alongside the bot-managed
        ``knowledge_base`` / ``prompt_resolver`` / ``prompt_envelope``
        collaborators). Collisions on those bot-managed names raise
        ``ConfigurationError`` to surface configuration errors loudly
        rather than silently dropping the consumer's value.
        """
        from dataknobs_llm.prompts import AsyncPromptBuilder
        from dataknobs_llm.prompts.implementations import CompositePromptLibrary

        from ..memory import create_memory_from_config

        llm = self.llm

        # Validate capability requirements (Layer 2 — startup check)
        # Only check main LLM requirements here; extraction LLM requirements
        # are validated when WizardReasoning sets up its extractor.
        from .validation import infer_main_capability_requirements

        requirements = infer_main_capability_requirements(
            {
                "reasoning": self.config.reasoning or {},
                "tools": self.config.tools,
                "llm": self.config.llm,
            }
        )
        if requirements:
            capabilities = llm.get_capabilities()
            capability_values = {cap.value for cap in capabilities}
            missing = [r for r in requirements if r not in capability_values]
            if missing:
                model_name = self.config.llm.get("model", "unknown")
                raise ConfigurationError(
                    f"Bot requires capabilities {missing} but model "
                    f"'{model_name}' provides "
                    f"{sorted(capability_values)}. "
                    f"Use a model that supports {missing} or "
                    f"update the environment resource configuration."
                )

        # Create conversation storage
        storage_config = dict(self.config.conversation_storage)
        storage_class_path = storage_config.pop("storage_class", None)
        has_backend = "backend" in storage_config

        if storage_class_path and has_backend:
            logger.warning(
                "Both 'backend' and 'storage_class' specified in "
                "conversation_storage. 'storage_class' takes precedence; "
                "'backend' will be ignored."
            )
        if not storage_class_path and not has_backend:
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

        # Build composed prompt library with precedence:
        #   1. Inline prompts (config.prompts) — highest priority
        #   2. Configured prompt_libraries — consumer file/config overrides
        #   3. Bots default library — all built-in prompt fragments
        #   4. Extraction default library — extraction prompts (lowest)
        from dataknobs_llm.prompts.implementations import ConfigPromptLibrary

        composed_libraries = []
        library_names = []

        # 1. Inline prompts from config (highest priority)
        if self.config.prompts is not None:
            prompts_config = self.config.prompts

            if isinstance(prompts_config, dict):
                structured_config = {"system": {}, "user": {}}

                for prompt_name, prompt_content in prompts_config.items():
                    if isinstance(prompt_content, dict):
                        prompt_type = prompt_content.get("type", "system")
                        if prompt_type in structured_config:
                            structured_config[prompt_type][prompt_name] = prompt_content
                    else:
                        structured_config["system"][prompt_name] = {
                            "template": prompt_content
                        }

                composed_libraries.append(ConfigPromptLibrary(structured_config))
                library_names.append("inline_config")

        # 2. Configured prompt_libraries (consumer overrides)
        if self.config.prompt_libraries:
            for lib_config in sorted(
                self.config.prompt_libraries,
                key=lambda c: c.get("priority", 50),
            ):
                lib_type = lib_config.get("type", "config")
                if lib_type == "filesystem":
                    from dataknobs_llm.prompts import FileSystemPromptLibrary

                    lib_path = lib_config.get("path", "")
                    composed_libraries.append(
                        FileSystemPromptLibrary(Path(lib_path))
                    )
                    library_names.append(f"filesystem:{lib_path}")
                elif lib_type == "config":
                    lib_prompts = lib_config.get("prompts", {})
                    structured = {"system": {}}
                    for name, content in lib_prompts.items():
                        if isinstance(content, dict):
                            structured["system"][name] = content
                        else:
                            structured["system"][name] = {"template": content}
                    composed_libraries.append(ConfigPromptLibrary(structured))
                    library_names.append("config_library")
                else:
                    logger.warning(
                        "Unknown prompt_library type %r — skipping",
                        lib_type,
                    )

        # 3. Bots default library (built-in prompt fragments)
        from dataknobs_bots.prompts.defaults import get_default_prompt_library

        composed_libraries.append(get_default_prompt_library())
        library_names.append("bots_defaults")

        # 4. Extraction default library (lowest priority)
        from dataknobs_llm.extraction.prompts import get_extraction_prompt_library

        composed_libraries.append(get_extraction_prompt_library())
        library_names.append("extraction_defaults")

        library = CompositePromptLibrary(
            libraries=composed_libraries, names=library_names,
        )
        prompt_builder = AsyncPromptBuilder(library)

        # Build prompt resolver for downstream components
        from dataknobs_bots.prompts.resolver import PromptResolver

        prompt_resolver = PromptResolver(library)

        # Create memory (pass llm so summary memory can use it)
        memory = None
        if self.config.memory is not None:
            memory = await create_memory_from_config(
                self.config.memory, llm_provider=llm,
                prompt_resolver=prompt_resolver,
            )

        # Create knowledge base BEFORE tools — tools may declare a
        # dependency on knowledge_base via catalog_metadata().requires
        knowledge_base = None
        kb_config = self.config.knowledge_base or {}
        kb_auto_context = kb_config.get("auto_context", True)
        if kb_config.get("enabled"):
            from ..knowledge import create_knowledge_base_from_config

            logger.info(
                "Initializing knowledge base with config: %s",
                kb_config.get("type", "unknown"),
            )
            knowledge_base = await create_knowledge_base_from_config(kb_config)
            logger.info("Knowledge base initialized successfully")

        # Build dependency map for tool injection
        tool_dependencies: dict[str, Any] = {}
        if knowledge_base is not None:
            tool_dependencies["knowledge_base"] = knowledge_base

        # Create tools (after KB so dependencies can be injected). Tool xref
        # resolution reads only ``tool_definitions`` from the surrounding
        # config, so a minimal context dict is sufficient.
        tool_ctx = {"tool_definitions": self.config.tool_definitions}
        tool_registry = ToolRegistry()
        if self.config.tools:
            for tool_config in self.config.tools:
                tool = self._resolve_tool(
                    tool_config, tool_ctx, tool_dependencies or None
                )
                if tool:
                    tool_registry.register_tool(tool)

        # Create reasoning strategy
        reasoning_strategy = None
        if self.config.reasoning is not None:
            from ..reasoning import create_reasoning_from_config

            reasoning_config = self.config.reasoning
            # Propagate config_base_path to reasoning if set at bot level
            config_base_path = self.config.config_base_path
            if config_base_path is not None:
                if "config_base_path" not in reasoning_config:
                    reasoning_config = {
                        **reasoning_config,
                        "config_base_path": config_base_path,
                    }
                elif reasoning_config["config_base_path"] != config_base_path:
                    logger.debug(
                        "Reasoning config has its own config_base_path=%r; "
                        "ignoring bot-level config_base_path=%r",
                        reasoning_config["config_base_path"],
                        config_base_path,
                    )
            extra_components = dict(reasoning_components or {})
            managed = {"knowledge_base", "prompt_resolver", "prompt_envelope"}
            collisions = managed & extra_components.keys()
            if collisions:
                raise ConfigurationError(
                    f"reasoning_components cannot override bot-managed "
                    f"component(s): {sorted(collisions)}. These are built "
                    f"by DynaBot itself from the bot's config "
                    f"(knowledge_base section, prompts section, "
                    f"prompt_envelope field) and forwarded into the "
                    f"strategy. Configure them through the bot config "
                    f"rather than overriding here."
                )
            reasoning_strategy = create_reasoning_from_config(
                reasoning_config,
                knowledge_base=knowledge_base,
                prompt_resolver=prompt_resolver,
                prompt_envelope=self._prompt_envelope,
                **extra_components,
            )

            # Config-driven source construction for strategies that
            # manage their own sources (grounded, hybrid, or any custom
            # strategy declaring manages_sources=True).
            strategy_caps = reasoning_strategy.capabilities()

            # Each strategy knows where its source configs live via
            # get_source_configs() — no hardcoded name checks needed.
            source_list = type(reasoning_strategy).get_source_configs(
                reasoning_config,
            )

            if strategy_caps.manages_sources and source_list:
                from ..knowledge.sources.factory import create_source_from_config
                from ..reasoning.grounded_config import GroundedSourceConfig

                for source_dict in source_list:
                    source_cfg = GroundedSourceConfig.from_dict(source_dict)
                    source = await create_source_from_config(
                        source_cfg,
                        knowledge_base=knowledge_base,
                    )
                    reasoning_strategy.add_source(source)

            # Auto-disable auto_context for source-managing strategies —
            # retrieval is structural and auto_context is completely
            # redundant.  Leaving it on causes KB-augmented messages as
            # query-generation input, wasting tokens and risking
            # timeouts with thinking models.
            if (
                strategy_caps.manages_sources
                and knowledge_base is not None
                and kb_auto_context
            ):
                kb_auto_context = False
                logger.info(
                    "%s: auto_context disabled "
                    "(retrieval is structural).",
                    type(reasoning_strategy).__name__,
                )

        # Create middleware
        if middleware_override is not None:
            middleware = list(middleware_override)
        else:
            middleware = []
            if self.config.middleware:
                for mw_config in self.config.middleware:
                    mw = self._create_bot_middleware(mw_config)
                    if mw:
                        middleware.append(mw)

        # Create conversation_middleware (ConversationMiddleware — LLM-call
        # wraps). Built the same way as ``middleware`` but lives in a
        # separate list because the two are forwarded to different layers:
        # ``self.middleware`` runs at bot-turn boundaries; this list is
        # forwarded to every ``ConversationManager`` this bot creates, so
        # it wraps the ``llm.complete`` call itself.
        if conversation_middleware_override is not None:
            conversation_middleware: list[ConversationMiddleware] = list(
                conversation_middleware_override
            )
        else:
            conversation_middleware = []
            if self.config.conversation_middleware:
                for mw_config in self.config.conversation_middleware:
                    mw = self._create_conversation_middleware(mw_config)
                    if mw is not None:
                        conversation_middleware.append(mw)

        # Extract system prompt (supports template name or inline content)
        system_prompt_name = None
        system_prompt_content = None
        system_prompt_rag_configs = None
        if self.config.system_prompt is not None:
            system_prompt_config = self.config.system_prompt
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

        # The effective context_transform (dotted path → callable) was
        # resolved in _setup; the scalar knobs (default temperature /
        # max tokens, tool iteration / timeout limits) were also derived
        # there from self.config.

        # Bind the built collaborators onto self.
        self.prompt_builder = prompt_builder
        self.conversation_storage = conversation_storage
        self.tool_registry = tool_registry
        self.memory = memory
        self.knowledge_base = knowledge_base
        self._kb_auto_context = kb_auto_context
        self.reasoning_strategy = reasoning_strategy
        self.middleware = middleware
        self._conversation_middleware = conversation_middleware
        # Config-driven build: the bot constructed these subsystems, so it
        # owns and closes them. (The pre-built / from_components path leaves
        # the _setup defaults False — those collaborators are caller-owned.)
        self._owns_conversation_storage = True
        self._owns_memory = True
        self._owns_knowledge_base = True
        self._owns_reasoning_strategy = True
        self.system_prompt_name = system_prompt_name
        self.system_prompt_content = system_prompt_content
        self.system_prompt_rag_configs = system_prompt_rag_configs
        self._prompt_resolver = prompt_resolver

        # Collect subsystem providers for catalog registration.
        # Each subsystem declares its own providers via providers().
        if memory is not None:
            for role, provider in memory.providers().items():
                self.register_provider(role, provider)

        if knowledge_base is not None:
            for role, provider in knowledge_base.providers().items():
                self.register_provider(role, provider)

        if reasoning_strategy is not None:
            for role, provider in reasoning_strategy.providers().items():
                self.register_provider(role, provider)

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
        *,
        add_observations: bool = True,
        executions_out: list[ToolExecution] | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> None:
        """Execute tool calls and optionally add observations to the conversation.

        Builds a ``ToolExecutionContext`` from the conversation manager,
        executes each tool, records ``ToolExecution`` on the turn state
        (or ``executions_out`` if provided), and optionally adds tool
        result observations to the conversation history.

        Args:
            turn: Current turn state (tool executions are appended here
                unless ``executions_out`` is provided).
            tool_calls: Tool call objects to execute.  Accepts LLM
                ``ToolCall`` objects (ReAct) or ``ToolCallSpec`` objects
                (wizard config-driven).  Both must have ``.name`` and
                ``.parameters`` attributes.
            add_observations: When ``False``, skip adding tool result
                messages to conversation history.  Used by the phased
                protocol for strategy-driven tool calls whose results
                flow through wizard state, not conversation history.
            executions_out: When provided, ``ToolExecution`` records are
                appended here instead of ``turn.tool_executions``.  Lets
                the phased path collect results without coupling to
                ``TurnState``.
            extra_context: Extra key-value pairs merged into the
                ``ToolExecutionContext`` via ``with_extra()``.  Used by
                strategies (e.g. ReAct) that inject artifact registries,
                review executors, or other infrastructure into tool calls.
        """
        target_list = executions_out if executions_out is not None else turn.tool_executions
        for tool_call in tool_calls:
            tool_name = tool_call.name
            tool_context = ToolExecutionContext.from_manager(turn.manager)
            if extra_context:
                tool_context = tool_context.with_extra(**extra_context)
            try:
                tool = self.tool_registry.get_tool(tool_name)
            except NotFoundError:
                tool = None

            if tool is None:
                observation = "Tool not found"
                target_list.append(ToolExecution(
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
                    # Route through the registry's ``execute_tool`` so
                    # the registry's ``_execution_tracker`` records the
                    # call when ``track_executions=True``.  Direct
                    # ``tool.execute`` would bypass the recording path
                    # — context-builder consumers of
                    # ``tool_registry.get_execution_history`` would see
                    # an empty list on real bot turns.  The registry
                    # forwards ``_context`` only to tools whose
                    # ``execute`` accepts ``**kwargs`` (e.g.
                    # ``ContextAwareTool``); plain Tools are unaffected.
                    result = await asyncio.wait_for(
                        self.tool_registry.execute_tool(
                            tool_name,
                            **tool_call.parameters,
                            _context=tool_context,
                        ),
                        timeout=self._tool_timeout,
                    )
                    duration_ms = (time.monotonic() - t0) * 1000
                    try:
                        observation = (
                            f"Tool result: {json.dumps(result, default=str)}"
                        )
                    except (TypeError, ValueError):
                        observation = f"Tool result: {result}"

                    target_list.append(ToolExecution(
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
                except (TimeoutError, asyncio.TimeoutError):
                    duration_ms = (time.monotonic() - t0) * 1000
                    observation = (
                        f"Error: tool timed out after "
                        f"{self._tool_timeout:.1f}s"
                    )
                    target_list.append(ToolExecution(
                        tool_name=tool_name,
                        parameters=tool_call.parameters,
                        error=(
                            f"Timed out after {self._tool_timeout:.1f}s"
                        ),
                        duration_ms=duration_ms,
                    ))
                    logger.warning(
                        "Tool execution timed out: %s (%.1fs limit)",
                        tool_name,
                        self._tool_timeout,
                        extra={
                            "conversation_id": getattr(
                                turn.manager, "conversation_id", None
                            ),
                            "duration_ms": round(duration_ms, 1),
                        },
                    )
                except Exception as exc:
                    observation = f"Error: {exc!s}"
                    target_list.append(ToolExecution(
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

            if add_observations:
                await turn.manager.add_message(
                    content=f"Observation from {tool_name}: {observation}",
                    role="tool",
                    name=tool_name,
                    tool_call_id=getattr(tool_call, "id", ""),
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
        # executions — ordering is by source, not chronological).
        # Three possible sources per turn (mutually exclusive):
        # - Phased ReAct: DynaBot's _execute_tools → turn.tool_executions
        #   (strategy._tool_executions stays empty)
        # - Non-phased ReAct (via generate()): strategy._tool_executions
        # - Wizard phased: executions_out list merged at end of
        #   _generate_phased_response
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

    async def _generate_phased_response(
        self,
        turn: TurnState,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config_overrides: dict[str, Any] | None = None,
    ) -> Any:
        """Execute phased reasoning flow with optional tool interleaving.

        Used for strategies that implement
        :class:`~dataknobs_bots.reasoning.base.PhasedReasoningProtocol`.
        Splits the turn into begin_turn / process_input / finalize_turn
        phases, allowing DynaBot to execute tools between process_input
        and finalize_turn.

        Args:
            turn: Current turn state (manager must be set).
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.
            llm_config_overrides: Optional per-request LLM config overrides.

        Returns:
            LLM response object.
        """
        strategy = self.reasoning_strategy
        assert strategy is not None

        handle = await strategy.begin_turn(  # type: ignore[union-attr]
            turn.manager,
            self.llm,
            tools=list(self.tool_registry) or None,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
            llm_config_overrides=llm_config_overrides,
        )

        if handle.early_response:
            return handle.early_response

        early_response, tool_results = await self._run_phased_process_loop(
            strategy, handle, turn
        )
        if early_response is not None:
            return early_response

        response = await strategy.finalize_turn(  # type: ignore[union-attr]
            handle, tool_results
        )

        # Merge phased tool executions into turn state so _finalize_turn
        # dispatches on_tool_executed middleware for these executions.
        if tool_results:
            turn.tool_executions.extend(tool_results)

        return response

    async def _run_phased_process_loop(
        self,
        strategy: Any,
        handle: Any,
        turn: TurnState,
    ) -> tuple[Any | None, list[ToolExecution] | None]:
        """Run the iterative process_input loop for phased strategies.

        Shared core for ``_generate_phased_response`` and the streaming
        phased path in ``stream_chat``.  Handles iteration cap, timeout,
        tool execution dispatch, and the ``iterate`` loop signal.

        Args:
            strategy: Phased reasoning strategy instance.
            handle: Turn handle from ``begin_turn``.
            turn: Current turn state.

        Returns:
            Tuple of ``(early_response, tool_results)``:
            - ``early_response`` is set when ``process_input`` returns
              an early response; the caller should return/yield it
              directly and skip ``finalize_turn``.
            - ``tool_results`` is a list of ``ToolExecution`` records
              from wizard-style tool calls (``add_observations=False``),
              or ``None``.  Passed to ``finalize_turn``.
        """
        # Iteration cap: strategy's declared max + 1 so the strategy
        # gets a final process_input call to detect its own cap and
        # return cleanly (e.g. writing trace entries).
        max_iters = (handle.max_iterations or self._max_tool_iterations) + 1
        tool_results: list[ToolExecution] | None = None
        loop_start = time.monotonic()
        result: ProcessResult | None = None
        for _iteration in range(max_iters):
            result = await strategy.process_input(handle)

            if result.early_response:
                return result.early_response, tool_results

            if not result.needs_tool_execution or not self.tool_registry:
                if result.needs_tool_execution and not self.tool_registry:
                    logger.warning(
                        "Strategy requested tool execution but no tools "
                        "are registered — skipping",
                    )
                break

            # Wall-clock timeout guard.  Add a system message so the LLM
            # knows the loop was truncated during the synthesis call.
            if time.monotonic() - loop_start >= self._tool_loop_timeout:
                logger.warning("Phased tool loop exceeded timeout")
                if result.iterate:
                    await turn.manager.add_message(
                        content=(
                            "System notice: The tool execution loop was "
                            "stopped due to a time limit. Please use the "
                            "results already available in the conversation "
                            "to respond to the user."
                        ),
                        role="system",
                    )
                break

            # Determine execution mode based on iterate flag.
            # Iterative strategies (ReAct) add observations to
            # conversation history; non-iterative (wizard) route results
            # through state, not history.
            if result.iterate:
                # ReAct: observations go into conversation history
                await self._execute_tools(
                    turn,
                    result.pending_tool_calls,
                    extra_context=handle.tool_extra_context if handle.tool_extra_context else None,
                )
            else:
                # Wizard: results flow through wizard state
                tool_results = []
                await self._execute_tools(
                    turn,
                    result.pending_tool_calls,
                    add_observations=False,
                    executions_out=tool_results,
                )
                if not tool_results:
                    tool_results = None

            if not result.iterate:
                break
        else:
            if result is not None and result.needs_tool_execution:
                logger.warning("Phased tool loop reached max iterations")

        return None, tool_results

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

    async def _call_finally_turn_middleware(self, turn: TurnState) -> None:
        """Dispatch finally_turn to all middleware.

        Fires after every turn — on both success and error paths.
        Called from the ``finally`` block in ``chat()``,
        ``stream_chat()``, and ``greet()`` (normal path), and directly
        from the no-strategy early-exit path in ``greet()`` when
        ``plugin_data`` was provided.

        Observational hook — one failing middleware must not prevent
        others from running.  Errors are logged, then reported to all
        middleware via ``on_hook_error``.

        Args:
            turn: Turn state at the end of the pipeline.
        """
        for mw in self.middleware:
            try:
                await mw.finally_turn(turn)
            except Exception as exc:
                logger.exception(
                    "Middleware %s.finally_turn raised",
                    type(mw).__name__,
                )
                await self._call_on_hook_error_middleware(
                    "finally_turn", exc, turn.context
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
        plugin_data: dict[str, Any] | None = None,
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
            plugin_data: Optional dict to seed ``turn.plugin_data`` before
                      middleware runs.  Enables caller-managed lifecycle
                      patterns (e.g., passing a DB session handle that
                      middleware can use and ``finally_turn`` can close).
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
            plugin_data=plugin_data or {},
        )
        try:
            await self._prepare_turn(turn)

            # Branch on phased reasoning support.  Strategies that
            # implement PhasedReasoningProtocol (e.g. WizardReasoning)
            # use the three-phase flow which enables tool interleaving.
            # All other strategies use the existing single-call path.
            # Import deferred to avoid bot ↔ reasoning circular import.
            from ..reasoning.base import PhasedReasoningProtocol

            if isinstance(self.reasoning_strategy, PhasedReasoningProtocol):
                response = await self._generate_phased_response(
                    turn, temperature, max_tokens, llm_config_overrides
                )
            else:
                response = await self._generate_response(
                    turn.manager, temperature, max_tokens, llm_config_overrides
                )

                # DynaBot-level tool execution loop.  Strategies that handle
                # tool_calls internally (e.g. ReAct) return responses without
                # tool_calls, so this loop is a no-op for them.
                loop_start = time.monotonic()
                for _iteration in range(self._max_tool_iterations):
                    if (
                        not self.tool_registry
                        or not getattr(response, "tool_calls", None)
                    ):
                        break
                    if time.monotonic() - loop_start >= self._tool_loop_timeout:
                        logger.warning(
                            "Tool execution loop exceeded wall-clock timeout "
                            "(%.1fs)",
                            self._tool_loop_timeout,
                            extra={
                                "conversation_id": getattr(
                                    turn.manager, "conversation_id", None
                                ),
                            },
                        )
                        break
                    await self._execute_tools(turn, response.tool_calls)
                    # Accumulate usage from intermediate LLM calls
                    turn.accumulate_usage(response)
                    # Enforce remaining loop budget on the LLM re-call
                    remaining = self._tool_loop_timeout - (
                        time.monotonic() - loop_start
                    )
                    if remaining <= 0:
                        logger.warning(
                            "Tool loop budget exhausted before LLM re-call "
                            "(%.1fs budget)",
                            self._tool_loop_timeout,
                            extra={
                                "conversation_id": getattr(
                                    turn.manager, "conversation_id", None
                                ),
                            },
                        )
                        break
                    try:
                        response = await asyncio.wait_for(
                            turn.manager.complete(
                                tools=list(self.tool_registry) or None,
                                temperature=temperature or self.default_temperature,
                                max_tokens=max_tokens or self.default_max_tokens,
                                llm_config_overrides=llm_config_overrides,
                            ),
                            timeout=remaining,
                        )
                    except (TimeoutError, asyncio.TimeoutError):
                        logger.warning(
                            "LLM re-call exceeded remaining tool loop "
                            "budget (%.1fs remaining of %.1fs)",
                            remaining,
                            self._tool_loop_timeout,
                            extra={
                                "conversation_id": getattr(
                                    turn.manager, "conversation_id", None
                                ),
                            },
                        )
                        break
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
        finally:
            await self._call_finally_turn_middleware(turn)

    async def greet(
        self,
        context: BotContext,
        *,
        initial_context: dict[str, Any] | None = None,
        plugin_data: dict[str, Any] | None = None,
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
            plugin_data: Optional dict to seed ``turn.plugin_data`` before
                middleware runs.  See ``chat()`` for details.

                When ``reasoning_strategy`` is ``None``, no turn is
                initiated but ``finally_turn`` still fires if
                ``plugin_data`` was provided, ensuring cleanup.

        Returns:
            Greeting string, or None if the bot does not support greetings

        Note:
            Middleware lifecycle for greet: ``on_turn_start(turn)`` and
            ``before_message("")`` are called before greeting generation;
            ``after_turn(turn)`` and ``after_message(...)`` are called on
            success (only when a response is generated);
            ``finally_turn(turn)`` fires on success, error, and when
            the strategy returns ``None`` (no greeting).
            If an error occurs, ``on_error`` hooks receive
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
            if plugin_data is not None:
                turn = TurnState(
                    mode=TurnMode.GREET,
                    message="",
                    context=context,
                    plugin_data=plugin_data,
                )
                await self._call_finally_turn_middleware(turn)
            return None

        turn = TurnState(
            mode=TurnMode.GREET,
            message="",
            context=context,
            initial_context=initial_context,
            plugin_data=plugin_data or {},
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
        finally:
            await self._call_finally_turn_middleware(turn)

    async def stream_chat(
        self,
        message: str,
        context: BotContext,
        temperature: float | None = None,
        max_tokens: int | None = None,
        rag_query: str | None = None,
        llm_config_overrides: dict[str, Any] | None = None,
        plugin_data: dict[str, Any] | None = None,
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
            plugin_data: Optional dict to seed ``turn.plugin_data`` before
                      middleware runs.  See ``chat()`` for details.
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

            **Cleanup guarantee:** ``finally_turn`` middleware fires via a
            ``finally`` block inside the async generator.  In Python, async
            generator ``finally`` blocks execute only when the generator is
            fully consumed, explicitly closed (``await gen.aclose()``), or
            garbage collected.  Callers that break out of the stream early
            should use ``contextlib.aclosing`` to guarantee prompt cleanup::

                from contextlib import aclosing

                async with aclosing(bot.stream_chat("msg", ctx)) as stream:
                    async for chunk in stream:
                        if done:
                            break  # aclose() fires finally_turn
        """
        turn = TurnState(
            mode=TurnMode.STREAM,
            message=message,
            context=context,
            rag_query=rag_query,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_config_overrides=llm_config_overrides,
            plugin_data=plugin_data or {},
        )
        streaming_error: Exception | None = None
        stream_fully_consumed = False

        try:
            await self._prepare_turn(turn)

            # Track tool_calls across streaming rounds so the tool
            # execution loop can pick them up after the initial stream.
            pending_tool_calls: list[Any] | None = None

            # Branch on phased reasoning support (same as chat()).
            # Import deferred to avoid bot ↔ reasoning circular import.
            # isinstance ordering matters: StreamingPhasedProtocol
            # extends PhasedReasoningProtocol, so check it first.
            from ..reasoning.base import (
                PhasedReasoningProtocol,
                StreamingPhasedProtocol,
            )

            if isinstance(self.reasoning_strategy, StreamingPhasedProtocol):
                # Streaming phased flow: begin_turn + process_input
                # blocking, stream finalize_turn per-token.
                strategy = self.reasoning_strategy
                handle = await strategy.begin_turn(
                    turn.manager,
                    self.llm,
                    tools=list(self.tool_registry) or None,
                    temperature=temperature or self.default_temperature,
                    max_tokens=max_tokens or self.default_max_tokens,
                    llm_config_overrides=llm_config_overrides,
                )

                if handle.early_response:
                    content = self._extract_response_content(
                        handle.early_response
                    )
                    turn.stream_chunks.append(content)
                    turn.populate_from_response(
                        handle.early_response, self.llm
                    )
                    yield LLMStreamResponse(
                        delta=content,
                        is_final=True,
                        finish_reason="stop",
                    )
                else:
                    early_response, tool_results = (
                        await self._run_phased_process_loop(
                            strategy, handle, turn
                        )
                    )

                    if early_response is not None:
                        content = self._extract_response_content(
                            early_response
                        )
                        turn.stream_chunks.append(content)
                        turn.populate_from_response(
                            early_response, self.llm
                        )
                        yield LLMStreamResponse(
                            delta=content,
                            is_final=True,
                            finish_reason="stop",
                        )
                    else:
                        # Stream finalize_turn
                        async for chunk in strategy.stream_finalize_turn(
                            handle, tool_results
                        ):
                            turn.stream_chunks.append(chunk.delta)
                            if chunk.is_final or chunk.usage:
                                turn.populate_from_final_stream_chunk(
                                    chunk, self.llm
                                )
                            yield chunk

                        # Merge phased tool executions into turn state
                        if tool_results:
                            turn.tool_executions.extend(tool_results)

            elif isinstance(self.reasoning_strategy, PhasedReasoningProtocol):
                # Non-streaming phased fallback (single chunk)
                response = await self._generate_phased_response(
                    turn, temperature, max_tokens, llm_config_overrides
                )
                content = self._extract_response_content(response)
                turn.stream_chunks.append(content)
                turn.populate_from_response(response, self.llm)
                yield LLMStreamResponse(
                    delta=content,
                    is_final=True,
                    finish_reason="stop",
                )
            elif self.reasoning_strategy:
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
            loop_start = time.monotonic()
            for _iteration in range(self._max_tool_iterations):
                if not pending_tool_calls or not self.tool_registry:
                    break
                if time.monotonic() - loop_start >= self._tool_loop_timeout:
                    logger.warning(
                        "Streaming tool execution loop exceeded "
                        "wall-clock timeout (%.1fs)",
                        self._tool_loop_timeout,
                        extra={
                            "conversation_id": getattr(
                                turn.manager, "conversation_id", None
                            ),
                        },
                    )
                    break
                await self._execute_tools(turn, pending_tool_calls)
                # Accumulate usage from intermediate streaming rounds
                turn.accumulate_usage_from_stream()
                pending_tool_calls = None

                # Check remaining budget before starting LLM re-stream
                remaining = self._tool_loop_timeout - (
                    time.monotonic() - loop_start
                )
                if remaining <= 0:
                    logger.warning(
                        "Streaming tool loop budget exhausted before "
                        "LLM re-stream (%.1fs budget)",
                        self._tool_loop_timeout,
                        extra={
                            "conversation_id": getattr(
                                turn.manager, "conversation_id", None
                            ),
                        },
                    )
                    break

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

            stream_fully_consumed = True

        except Exception as e:
            streaming_error = e
            await self._call_on_error_middleware(e, message, context)
            raise
        finally:
            # Only finalize when the stream was fully consumed (not
            # on early exit via aclose/break, which would write
            # partial data to conversation history).
            if streaming_error is None and stream_fully_consumed:
                turn.response_content = "".join(turn.stream_chunks)
                await self._finalize_turn(turn)
            await self._call_finally_turn_middleware(turn)

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

    def get_steps_of_type(self, step_cls: type[_StepT]) -> list[_StepT]:
        """Return reasoning-strategy pipeline steps matching ``step_cls``.

        Iterates ``self.reasoning_strategy.steps`` (when the strategy
        exposes one) and filters by ``isinstance``. Returns an empty
        list when the bot has no reasoning strategy, when the strategy
        is not pipeline-shaped (no ``steps`` attribute), or when no
        step matches.

        Intended for post-construction injection of runtime collaborators
        that configuration cannot carry (e.g. resources owned by the
        host application's lifespan). The typed return removes the need
        for the caller to ``getattr``-chain through the strategy or to
        write an ``isinstance`` filter inline.

        Args:
            step_cls: The class to filter by. Subclass instances match.

        Returns:
            Matching steps in pipeline insertion order; empty list when
            no match is possible. The returned list is a snapshot —
            mutations to it do not affect the strategy's step collection.

        Example:
            ```python
            for step in bot.get_steps_of_type(MyHandler):
                step.attach_service(service)
            ```
        """
        strategy = self.reasoning_strategy
        if strategy is None:
            return []
        steps = getattr(strategy, "steps", None)
        if steps is None:
            return []
        return [s for s in steps if isinstance(s, step_cls)]

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

        Teardown is gated on ownership: this method closes only the
        collaborators the bot built itself (from config) — the main LLM
        provider, conversation storage backend, reasoning strategy,
        knowledge base, and memory store — releasing their associated
        resources like HTTP and database connections. A collaborator
        injected via ``from_components`` / the pre-built constructor (or
        ``from_config(llm=...)``) is caller-owned and left open, so a
        provider, KB, storage, memory, or strategy shared across bots
        survives one bot's close. Should be called when the bot is no
        longer needed, especially in testing or when creating temporary
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
        # Only subsystems this bot owns (built from config) are torn down;
        # a collaborator injected via from_components / the pre-built
        # constructor is caller-owned and left open, so a KB / storage /
        # memory / strategy shared across bots survives one bot's close.
        await close_if_owned(
            self.knowledge_base,
            self._owns_knowledge_base,
            on_error=lambda _exc: logger.exception("Error closing knowledge base"),
        )
        await close_if_owned(
            self.reasoning_strategy,
            self._owns_reasoning_strategy,
            on_error=lambda _exc: logger.exception(
                "Error closing reasoning strategy"
            ),
        )
        await close_if_owned(
            self.memory,
            self._owns_memory,
            on_error=lambda _exc: logger.exception("Error closing memory store"),
        )
        # Close conversation storage
        await close_if_owned(
            self.conversation_storage,
            self._owns_conversation_storage,
            on_error=lambda _exc: logger.exception(
                "Error closing conversation storage"
            ),
        )
        # Close main LLM provider only if DynaBot created it.
        # When from_config(llm=...) was used, the caller owns the lifecycle.
        await close_if_owned(
            self.llm,
            self._owns_llm,
            on_error=lambda _exc: logger.exception(
                "Error closing main LLM provider"
            ),
        )

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
                middleware=list(self._conversation_middleware),
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
                middleware=list(self._conversation_middleware),
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
            Message augmented with context, wrapped in the style chosen
            by ``DynaBotConfig.prompt_envelope`` (default markdown). See
            :class:`~dataknobs_bots.prompts.PromptEnvelope`.
        """
        envelope = self._prompt_envelope
        sections: list[str] = []

        # Knowledge context (skip when auto_context is disabled — KB
        # remains available for tool-based access). Ask the KB layer
        # for the body text and let the envelope decide the wrapper
        # shape; this keeps the wrap decision in one place instead of
        # duplicating it inside every KnowledgeBase implementation.
        if self.knowledge_base and self._kb_auto_context:
            search_query = rag_query if rag_query else message
            kb_results = await self.knowledge_base.query(search_query, k=5)
            if kb_results:
                kb_body = self.knowledge_base.format_context(
                    kb_results, wrap_in_tags=False
                )
                if self._context_transform:
                    kb_body = self._context_transform(kb_body)
                sections.append(envelope.knowledge_base_section(kb_body))

        # Conversation history context.
        if self.memory:
            mem_results = await self.memory.get_context(message)
            if mem_results:
                mem_body = "\n\n".join(r["content"] for r in mem_results)
                if self._context_transform:
                    mem_body = self._context_transform(mem_body)
                sections.append(envelope.conversation_history_section(mem_body))

        # No context sections → return the bare message. Wrapping a
        # lone question with no surrounding context adds no signal and
        # would surprise direct callers whose configs have neither KB
        # auto-context nor memory.
        if not sections:
            return message

        # Skip the question section when the message is empty. An
        # empty body would render to ``""`` (the envelope's empty-body
        # contract) and join with a trailing joiner separator,
        # producing a malformed prompt ending in ``"\n\n---\n\n"``
        # (markdown) or ``"\n\n"`` (xml/prose). No real caller passes
        # an empty message, but the guard keeps the output well-formed
        # for the contrived (e.g. RAG-query-only) case.
        if message:
            sections.append(envelope.question_section(message))
        return envelope.joiner().join(sections)

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
            tool_config: Tool configuration (dict or string xref).
                Dict configs support an ``optional`` key (default False).
                When ``optional: true``, transient resolution failures
                (missing module / class, ``from_config`` raising, ctor
                raising on bad params) log a warning and return ``None``
                instead of raising. ``optional: true`` does NOT cover
                class-shape mismatches — a resolved class that does not
                subclass ``Tool`` always raises, because that is a
                programmer error in the config layout (wrong dotted
                path, spec listed under the wrong field), not a
                transient environment failure. When ``optional`` is
                ``False`` (default), every failure raises
                ``ConfigurationError``.
            config: Full bot configuration for xref resolution
            dependencies: Optional resource dependencies to inject into tools
                that declare them via catalog_metadata().requires

        Returns:
            Tool instance, or ``None`` if a transient resolution failure
            occurred and ``optional: true`` was set.

        Raises:
            ConfigurationError: If the tool cannot be resolved and is
                not marked ``optional: true``, OR if the resolved class
                is not a subclass of ``Tool`` (always raises, regardless
                of ``optional``).

        Example:
            # Direct instantiation (required — fails loudly)
            tool_config = {
                "class": "my_tools.CalculatorTool",
                "params": {"precision": 2}
            }

            # Optional tool — skipped if unavailable
            tool_config = {
                "class": "my_tools.OptionalTool",
                "optional": true,
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

        optional = (
            tool_config.get("optional", False)
            if isinstance(tool_config, dict)
            else False
        )

        try:
            # Handle xref string format
            if isinstance(tool_config, str):
                if tool_config.startswith("xref:"):
                    import re

                    match = re.match(r"xref:tools\[([^\]]+)\]", tool_config)
                    if not match:
                        if optional:
                            logger.warning("Skipping optional tool: Invalid xref format: %s", tool_config)
                            return None
                        raise ConfigurationError(f"Invalid xref format: {tool_config}")

                    tool_name = match.group(1)

                    tool_definitions = config.get("tool_definitions", {})
                    if tool_name not in tool_definitions:
                        msg = (
                            f"Tool definition not found: {tool_name}. "
                            f"Available: {list(tool_definitions.keys())}"
                        )
                        if optional:
                            logger.warning("Skipping optional tool: %s", msg)
                            return None
                        raise ConfigurationError(msg)

                    # Propagate optional into the resolved definition so
                    # that import/instantiation errors honour the flag.
                    resolved = tool_definitions[tool_name]
                    if optional and isinstance(resolved, dict) and not resolved.get("optional"):
                        resolved = {**resolved, "optional": True}
                    return DynaBot._resolve_tool(resolved, config, dependencies)
                else:
                    if optional:
                        logger.warning(
                            "Skipping optional tool: String tool config must be xref format: %s",
                            tool_config,
                        )
                        return None
                    raise ConfigurationError(
                        f"String tool config must be xref format: {tool_config}"
                    )

            # Handle dict with xref key — resolve the referenced string,
            # injecting optional into the resolved definition if set.
            if isinstance(tool_config, dict) and "xref" in tool_config:
                xref_str = tool_config["xref"]
                if not optional:
                    return DynaBot._resolve_tool(xref_str, config, dependencies)
                # optional=True: resolve the xref string inline so we
                # can inject optional into the resolved definition
                # (the string path recomputes optional=False for strings).
                import re

                match = re.match(r"xref:tools\[([^\]]+)\]", xref_str) if isinstance(xref_str, str) else None
                if not match:
                    logger.warning("Skipping optional tool: Invalid xref format: %s", xref_str)
                    return None
                tool_name = match.group(1)
                tool_definitions = config.get("tool_definitions", {})
                if tool_name not in tool_definitions:
                    logger.warning(
                        "Skipping optional tool: Tool definition not found: %s. Available: %s",
                        tool_name,
                        list(tool_definitions.keys()),
                    )
                    return None
                resolved = tool_definitions[tool_name]
                if isinstance(resolved, dict) and not resolved.get("optional"):
                    resolved = {**resolved, "optional": True}
                return DynaBot._resolve_tool(resolved, config, dependencies)

            # Handle dict with class key (direct instantiation)
            if isinstance(tool_config, dict) and "class" in tool_config:
                class_path = tool_config["class"]
                params = dict(tool_config.get("params", {}))

                # Import the tool class
                module_path, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                tool_class = getattr(module, class_name)

                # Class-shape check BEFORE instantiation / from_config.
                # A wrong-shape class is a programmer error (the spec is
                # listed under the wrong field, or the dotted path points
                # at the wrong symbol entirely), not a transient
                # environment failure. ``optional: true`` is therefore
                # NOT honored here — it covers missing modules / classes
                # / bad params, not a class that resolved successfully
                # but is the wrong type. Running ``from_config`` or the
                # constructor before the shape check would let a
                # misplaced spec trigger ctor side effects (network
                # reads, file opens, log writes); mirroring the
                # middleware-helper policy keeps that closed.
                from dataknobs_llm.tools import Tool

                if not (
                    isinstance(tool_class, type) and issubclass(tool_class, Tool)
                ):
                    raise ConfigurationError(
                        f"Resolved class {class_path} must subclass "
                        f"{Tool.__module__}.{Tool.__qualname__} — got "
                        f"{tool_class!r}."
                    )

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

                logger.info("Successfully loaded tool: %s (%s)", tool.name, class_path)
                return tool

            msg = (
                f"Invalid tool config format. Expected dict with "
                f"'class' or 'xref' key, or xref string. "
                f"Got: {type(tool_config)}"
            )
            if optional:
                logger.warning("Skipping optional tool: %s", msg)
                return None
            raise ConfigurationError(msg)

        except ConfigurationError:
            raise
        except ImportError as e:
            msg = f"Failed to import tool class: {e}"
            if optional:
                logger.warning("Skipping optional tool: %s", msg)
                return None
            raise ConfigurationError(msg) from e
        except AttributeError as e:
            msg = f"Failed to find tool class: {e}"
            if optional:
                logger.warning("Skipping optional tool: %s", msg)
                return None
            raise ConfigurationError(msg) from e
        except Exception as e:
            msg = f"Failed to instantiate tool: {e}"
            if optional:
                logger.warning("Skipping optional tool: %s", msg)
                return None
            raise ConfigurationError(msg) from e

    @staticmethod
    def _resolve_middleware_from_spec(
        config: dict[str, Any],
        expected_base: type,
        *,
        label: str,
    ) -> Any | None:
        """Resolve a middleware spec to an instance, validating its class shape.

        Shared resolution body behind :meth:`_create_bot_middleware`
        (bot-turn :class:`~dataknobs_bots.middleware.Middleware`) and
        :meth:`_create_conversation_middleware` (LLM-call
        :class:`~dataknobs_llm.conversations.ConversationMiddleware`). The
        two flavors are wired to different layers but share this
        construction shape (``class`` + ``params`` + ``optional``).

        The class-shape check uses ``issubclass`` BEFORE instantiation so
        a wrong-shape spec never runs its ctor (avoiding network reads /
        file opens / log writes a misplaced spec's initializer might
        trigger). Type-mismatch errors raise unconditionally —
        ``optional: true`` covers transient resolution failures (module /
        class / params), NOT a class listed under the wrong field. A
        misplaced spec (a turn-lifecycle ``Middleware`` listed under
        ``conversation_middleware:``, or vice versa) is a programmer error
        in the config layout, and the only safe response is to surface it
        at config-load.

        Args:
            config: Middleware configuration dict with:
                - class: Dotted import path to middleware class
                - params: Optional constructor parameters
                - optional: If True, log warning and skip on resolution
                  failure (missing module / class / bad params) instead of
                  raising (default: False). Does NOT apply to class-shape
                  mismatches — those always raise.
            expected_base: Class the resolved middleware must subclass
                (``Middleware`` for bot turn-lifecycle hooks,
                ``ConversationMiddleware`` for LLM-call wraps).
            label: Human-readable label used in error / log messages
                (e.g. ``"middleware"``, ``"conversation_middleware"``).

        Returns:
            Instantiated middleware, or ``None`` if resolution fails
            (NOT a class-shape mismatch) and ``optional: true`` was set.

        Raises:
            ConfigurationError: If the class cannot be resolved or
                instantiation fails, unless ``optional: true``; OR if the
                resolved class is not a subclass of ``expected_base``
                (always raises, regardless of ``optional``).
        """
        import importlib

        optional = config.get("optional", False)
        class_path = config.get("class", "<missing>")

        try:
            module_path, class_name = config["class"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            middleware_class = getattr(module, class_name)
        except Exception as e:
            # Resolution failure (missing module / class / malformed spec)
            # — covered by ``optional``.
            msg = f"Failed to resolve {label} '{class_path}': {e}"
            if optional:
                logger.warning("Skipping optional %s: %s", label, msg)
                return None
            raise ConfigurationError(msg) from e

        # Class-shape check BEFORE instantiation. Never optional: a class
        # listed under the wrong field is a programmer error, not a
        # transient environment failure.
        if not (
            isinstance(middleware_class, type)
            and issubclass(middleware_class, expected_base)
        ):
            raise ConfigurationError(
                f"{label} '{class_path}' must subclass "
                f"{expected_base.__module__}.{expected_base.__qualname__} "
                f"— check whether this spec belongs under 'middleware' "
                f"(bot-turn hooks) or 'conversation_middleware' "
                f"(LLM-call wraps)."
            )

        try:
            return middleware_class(**dict(config.get("params", {})))
        except Exception as e:
            # Instantiation failure (bad params, ctor raised) — covered by
            # ``optional``.
            msg = f"Failed to instantiate {label} '{class_path}': {e}"
            if optional:
                logger.warning("Skipping optional %s: %s", label, msg)
                return None
            raise ConfigurationError(msg) from e

    @staticmethod
    def _create_bot_middleware(config: dict[str, Any]) -> Middleware | None:
        """Resolve a bot-turn ``Middleware`` spec.

        See :meth:`_resolve_middleware_from_spec` for the resolution,
        class-shape validation, and ``optional`` semantics.
        """
        return DynaBot._resolve_middleware_from_spec(
            config, Middleware, label="middleware"
        )

    @staticmethod
    def _create_conversation_middleware(
        config: dict[str, Any],
    ) -> ConversationMiddleware | None:
        """Resolve a ``ConversationMiddleware`` (LLM-call wrap) spec.

        See :meth:`_resolve_middleware_from_spec` for the resolution,
        class-shape validation, and ``optional`` semantics.
        """
        return DynaBot._resolve_middleware_from_spec(
            config, ConversationMiddleware, label="conversation_middleware"
        )

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
        """Reinstate strategy state from a checkpoint node's metadata.

        Bot-side responsibility: locate the node and validate its data
        shape. Strategy-side responsibility: read its own keys out of
        the node's metadata and reinstate them. The method name retains
        its ``wizard`` historical anchor because that's the only strategy
        today that overrides ``restore_from_checkpoint``; the dispatch
        itself is strategy-agnostic.
        """
        if self.reasoning_strategy is None:
            return
        if manager.state is None:
            return

        node = get_node_by_id(manager.state.message_tree, node_id)
        if node is None:
            return

        node_data = node.data
        if not isinstance(node_data, ConversationNode):
            return

        self.reasoning_strategy.restore_from_checkpoint(
            manager, node_data.metadata
        )

    def _undo_banks_to_checkpoint(self, checkpoint_node_id: str) -> None:
        """Forward checkpoint-revert to the reasoning strategy.

        Strategies that hold node-keyed state (e.g. wizard memory banks)
        override ``ReasoningStrategy.undo_to_checkpoint``. Others inherit
        the base no-op.
        """
        if self.reasoning_strategy is None:
            return
        self.reasoning_strategy.undo_to_checkpoint(checkpoint_node_id)
