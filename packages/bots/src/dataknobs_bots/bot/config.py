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

#: Default text surfaced to the user when a phased reasoning turn's terminal
#: synthesis is cut off by the tool-loop wall-clock budget. Provider-neutral
#: and free of any consumer/product naming so it is a safe out-of-the-box
#: default; consumers localize / brand / soften it via
#: ``DynaBotConfig.tool_loop_timeout_message``.
_DEFAULT_TOOL_LOOP_TIMEOUT_MESSAGE = (
    "The response was cut short due to a time limit."
)


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
        middleware: Optional list of bot-level middleware specs
            (``dataknobs_bots.middleware.Middleware`` — turn-lifecycle hooks).
        conversation_middleware: Optional list of LLM-call-level middleware
            specs (``dataknobs_llm.conversations.ConversationMiddleware`` —
            request/response wraps around ``llm.complete``). Forwarded to
            every ``ConversationManager`` this bot constructs. Each spec is
            a dict with ``class`` (dotted import path) and optional
            ``params`` / ``optional`` keys, same shape as ``middleware``.
            Distinct from ``middleware`` because the two interfaces are
            structurally different (bot-turn hooks vs LLM-call wraps).
        system_prompt: Optional system-prompt configuration — a template
            name (smart-detected) or a dict with ``name`` / ``content`` /
            ``rag_configs`` / ``strict``.
        context_transform: Optional dotted import path to a callable applied
            to injected content (KB chunks, memory context) before it
            reaches the prompt. The callable form is supplied
            programmatically at construction time and is therefore not part
            of this serializable snapshot.
        prompt_envelope: Style used to wrap labeled context sections
            ("Knowledge base", "Conversation history", "Question") in
            the user prompt assembled by ``_build_message_with_context``
            and in the grounded-reasoning synthesis system prompt. One
            of ``"markdown"`` (default), ``"xml"`` (legacy shape kept as
            an opt-in escape hatch), or ``"prose"``. Case-insensitive —
            ``"XML"`` / ``"Markdown"`` are accepted and normalized to
            lowercase. See
            :class:`~dataknobs_bots.prompts.PromptEnvelopeStyle`.
        config_base_path: Optional base directory for resolving relative
            paths in nested configs (e.g. ``wizard_config``).
        max_tool_iterations: Maximum tool-execution rounds before returning.
        tool_timeout: Per-tool execution timeout in seconds.
        tool_loop_timeout: Wall-clock budget for the tool-execution loop.
            For phased reasoning strategies this also bounds the terminal
            synthesis (``finalize_turn`` / ``stream_finalize_turn``) that
            runs after an abnormal loop termination: the synthesis is
            granted the remaining budget (floored at a small minimum) and,
            on timeout, degrades to ``tool_loop_timeout_message`` rather
            than running unbounded.
        tool_loop_timeout_message: Text surfaced to the user when the phased
            terminal synthesis exceeds the remaining ``tool_loop_timeout``
            budget and is cut off. Overridable so consumers can localize /
            brand / soften the degraded response without subclassing. An
            empty string is a legitimate choice (yields empty content /
            an empty final chunk) and is not rejected.
    """

    # Adopt polymorphic-section validation for the subsystem sections whose
    # config families are registry-resolvable today. A
    # ``DynaBotConfig.from_dict(raw).validate()`` dry-run-builds the resolved
    # ``llm`` / ``memory`` / ``knowledge_base`` / ``reasoning`` configs to
    # surface field errors / an unknown discriminator (backend ``type`` /
    # provider / ``strategy``) at config-parse time (without constructing the
    # bot), and — because ``RAGKnowledgeBaseConfig`` carries its own
    # ``vector_store`` binding and the grounded/hybrid strategy configs carry
    # nested sub-config trees — descends into those nested sections too.
    # Bindings are strings: ``dataknobs-bots`` registers the ``memory`` /
    # ``knowledge_base`` / ``reasoning`` resolvers eagerly on import and
    # ``dataknobs-llm`` registers the ``llm`` resolver, so this adds no import
    # of a subsystem config type. ``conversation_storage`` is owned by
    # ``dataknobs-llm`` and remains unbound for now.
    _polymorphic_fields: ClassVar[Mapping[str, str]] = {
        "llm": "llm",
        "memory": "memory",
        "knowledge_base": "knowledge_base",
        "reasoning": "reasoning",
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
    conversation_middleware: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str | dict[str, Any] | None = None

    # --- scalars / misc ---
    context_transform: str | None = None
    prompt_envelope: str = "markdown"
    config_base_path: str | None = None
    max_tool_iterations: int = 5
    tool_timeout: float = 30.0
    tool_loop_timeout: float = 120.0
    tool_loop_timeout_message: str = _DEFAULT_TOOL_LOOP_TIMEOUT_MESSAGE

    def __post_init__(self) -> None:
        """Validate the timeout + prompt_envelope invariants against the snapshot.

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
        # Validate prompt_envelope is a known style. Case-insensitive:
        # YAML configs are human-written, so ``"XML"`` / ``"Markdown"``
        # should not be a parse error. The value is normalized to
        # lowercase on the frozen snapshot so downstream lookups
        # (``PromptEnvelopeStyle(self.config.prompt_envelope)``) match
        # the enum's lowercase values.
        #
        # The import is intentionally local — not because there is a
        # cycle today (``dataknobs_bots.prompts.envelope`` depends only
        # on stdlib), but as a precaution: ``bot.config`` is imported
        # very early in the bot lifecycle and a future back-edge in any
        # ``dataknobs_bots.prompts.*`` module (e.g. a prompt template
        # type-hinting ``DynaBotConfig``) would silently re-enter this
        # module mid-import and surface as a confusing
        # ``PartiallyInitializedModule`` error. Keeping the import
        # local makes that contract explicit. Python caches the import
        # after first call, so the per-construction cost is one dict
        # lookup.
        from dataknobs_bots.prompts.envelope import PromptEnvelopeStyle

        raw = self.prompt_envelope
        normalized = raw.lower() if isinstance(raw, str) else raw
        try:
            PromptEnvelopeStyle(normalized)
        except ValueError as exc:
            valid = ", ".join(repr(s.value) for s in PromptEnvelopeStyle)
            raise ValueError(
                f"prompt_envelope must be one of {valid} (case-insensitive), "
                f"got {raw!r}"
            ) from exc
        if normalized != raw:
            # Frozen dataclass — bypass the immutability for the
            # normalization. The caller's mapping/typed-config is not
            # mutated; this only touches the snapshot we own.
            object.__setattr__(self, "prompt_envelope", normalized)


__all__ = ["DynaBotConfig"]
