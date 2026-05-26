"""Typed top-level configuration for :class:`~dataknobs_bots.bot.base.DynaBot`.

``DynaBotConfig`` is the one typed configuration snapshot for a bot. It is
a deliberately thin envelope: a handful of typed scalars plus the
documented config sections forwarded verbatim to the subsystem factories.

The polymorphic subsystem sections (``memory``, ``knowledge_base``,
``reasoning``) and the provider section (``llm``) stay raw mappings. Their
concrete type is chosen by a discriminator key dispatched in the subsystem
registry layer, so rebuilding them declaratively from the static field
graph would cross the boundary that keeps discriminated selection in the
registries. They pass through :meth:`StructuredConfig.from_dict` untouched
and are handed to the factories during the bot's async build.

The free-form prompt, tool, and middleware sections are likewise kept as
mappings / lists of spec dicts — they are consumed by the existing inline
build logic, not rebuilt into typed trees here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from dataknobs_common.structured_config import StructuredConfig


@dataclass(frozen=True)
class DynaBotConfig(StructuredConfig):
    """Typed snapshot of a bot's construction parameters.

    Attributes:
        llm: Provider/model configuration (provider, model, temperature,
            max_tokens, ...). Raw mapping — LLM-provider config is owned
            by ``dataknobs-llm`` and the effective default temperature /
            max tokens are read from here. Optional when a pre-built
            provider is injected at construction time.
        conversation_storage: Storage configuration. Either ``backend``
            (a database backend key for the default storage) or
            ``storage_class`` (a dotted import path to a custom
            ``ConversationStorage``).
        memory: Optional memory configuration, dispatched by its ``type``
            discriminator in the memory backend registry.
        knowledge_base: Optional knowledge-base configuration, built only
            when ``enabled`` is set. ``auto_context`` controls automatic
            KB injection.
        reasoning: Optional reasoning-strategy configuration, dispatched by
            its ``strategy`` discriminator in the reasoning registry.
        prompts: Optional inline prompts library (highest-priority layer of
            the composed prompt library).
        prompt_libraries: Optional list of additional prompt-library specs
            (filesystem or config), merged by ``priority``.
        tools: Optional list of tool specs (class / xref / definition
            references) resolved against ``tool_definitions``.
        tool_definitions: Optional named tool definitions referenced by
            ``xref`` entries in ``tools``.
        middleware: Optional list of middleware specs.
        system_prompt: Optional system-prompt configuration — a template
            name (smart-detected) or a dict with ``name`` / ``content`` /
            ``rag_configs`` / ``strict``.
        context_transform: Optional dotted import path to a callable applied
            to injected content (KB chunks, memory context) before it
            reaches the prompt. The callable form is supplied
            programmatically at construction time and is therefore not part
            of this serializable snapshot.
        config_base_path: Optional base directory for resolving relative
            paths in nested configs (e.g. ``wizard_config``).
        max_tool_iterations: Maximum tool-execution rounds before returning.
        tool_timeout: Per-tool execution timeout in seconds.
        tool_loop_timeout: Wall-clock budget for the tool-execution loop.
    """

    # Adopt polymorphic-section validation for the subsystem sections whose
    # config families are registry-resolvable today. A
    # ``DynaBotConfig.from_dict(raw).validate()`` dry-run-builds the resolved
    # ``memory`` / ``knowledge_base`` configs to surface field errors / an
    # unknown backend ``type`` at config-parse time (without constructing the
    # bot), and — because ``RAGKnowledgeBaseConfig`` carries its own
    # ``vector_store`` binding — descends into the nested vector-store section
    # too. Bindings are strings: ``dataknobs-bots`` registers the ``memory`` /
    # ``knowledge_base`` resolvers eagerly on import, so this adds no import of
    # a subsystem config type. ``reasoning`` is intentionally NOT bound yet (it
    # needs a reasoning-strategy config family first); ``conversation_storage``
    # is owned by ``dataknobs-llm`` and likewise unbound for now.
    _polymorphic_fields: ClassVar[Mapping[str, str]] = {
        "memory": "memory",
        "knowledge_base": "knowledge_base",
    }

    # --- provider / storage (raw mappings) ---
    llm: dict[str, Any] = field(default_factory=dict)
    conversation_storage: dict[str, Any] = field(default_factory=dict)

    # --- polymorphic subsystem specs (raw; dispatched by the registries) ---
    memory: dict[str, Any] | None = None
    knowledge_base: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None

    # --- prompt / tool / middleware sections (free-form / list-of-spec) ---
    prompts: dict[str, Any] | None = None
    prompt_libraries: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_definitions: dict[str, Any] = field(default_factory=dict)
    middleware: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str | dict[str, Any] | None = None

    # --- scalars / misc ---
    context_transform: str | None = None
    config_base_path: str | None = None
    max_tool_iterations: int = 5
    tool_timeout: float = 30.0
    tool_loop_timeout: float = 120.0

    def __post_init__(self) -> None:
        """Validate the timeout invariants against the config snapshot.

        Both construction paths (dict-driven and pre-built collaborator)
        build a ``DynaBotConfig`` snapshot, so validating here covers them
        uniformly.
        """
        if self.tool_timeout < 0:
            raise ValueError(
                f"tool_timeout must be non-negative, got {self.tool_timeout}"
            )
        if self.tool_loop_timeout < 0:
            raise ValueError(
                f"tool_loop_timeout must be non-negative, got "
                f"{self.tool_loop_timeout}"
            )


__all__ = ["DynaBotConfig"]
