"""Configuration dataclass for the wizard reasoning strategy.

The wizard strategy's reasoning-section config differs from the other
strategies: most of its behavioural knobs live *inside* the wizard
definition (the ``wizard_config`` YAML/dict, under its ``settings:``
block — already typed by
:class:`~dataknobs_bots.config.wizard_builder.WizardConfig`).  What
remains at the reasoning-section layer is a thin envelope of
construction inputs read by :meth:`WizardReasoning.from_config`.

The polymorphic / opaque sub-sections (``wizard_config``,
``extraction_config``, ``hooks``, ``artifacts``, ``review_protocols``,
``custom_functions``) stay raw mappings here — each is dispatched to its
own builder by ``from_config`` and is not re-typed in this envelope,
mirroring the raw-polymorphic-section norm used elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dataknobs_common.structured_config import StructuredConfig


@dataclass(frozen=True)
class WizardReasoningConfig(StructuredConfig):
    """Typed envelope for the wizard reasoning-section config.

    Mirrors the keys read by :meth:`WizardReasoning.from_config`.
    ``from_dict``/``to_dict`` are inherited from :class:`StructuredConfig`;
    the opaque sub-sections remain raw dicts (or, for ``wizard_config``, a
    path string).  Credentials nested inside ``extraction_config`` are
    masked by the base repr's interior-key descent, so no field-level
    ``_SENSITIVE_FIELDS`` entry is required.

    Attributes:
        wizard_config: Path to a wizard YAML config file, or an inline
            wizard-definition dict (compatible with
            ``WizardConfigLoader.load_from_dict``).  Required.
        config_base_path: Base directory for resolving a relative
            ``wizard_config`` path.
        custom_functions: Custom transition functions (callables or
            ``"module:function"`` string references).
        extraction_config: Extraction provider configuration used to
            build a ``SchemaExtractor``.
        strict_validation: Whether to enforce schema validation.
        hooks: Lifecycle hook configuration.
        artifacts: Artifact type definitions and backend selection.
        review_protocols: Review protocol definitions.
        initial_data: Seed data merged into the wizard state at start.
        consistent_navigation_lifecycle: Whether back/skip fire the same
            lifecycle hooks as forward transitions.
        manager_metadata_inbox_key: Optional ``manager.metadata`` key
            (or list of keys) whose value (a flat dict) is popped at
            the start of each turn and merged into
            :attr:`WizardState.data`.  Implemented internally as an
            auto-registered ``on_turn_start`` hook (see
            :func:`dataknobs_bots.reasoning.wizard_inbox.make_metadata_inbox_hook`).

            Cross-turn bridge for per-stage sub-strategy output (e.g.
            pipeline step signals) into the wizard's transition-eval
            scope without widening the safe-eval scope.  ``None``
            (default) disables the bridge — zero behaviour change.

            Bridge contract:

            * **Consume-on-read** — keys are popped, not get'd. Stale
              signals can't leak.
            * **None-as-eviction** — inbox value ``{x: None}`` writes
              ``None`` into ``state.data[x]`` (with the default merge).
            * **Empty-dict no-op** — ``{}`` is silently skipped (key
              still popped per consume-on-read).
            * **Across-turn semantics** — turn N's writer publishes
              AFTER turn N's transition; turn N+1's ``begin_turn``
              consumes.
            * **Multi-key** — list-valued key drains all listed keys
              per turn.
            * **greet inherits** — bot-initiated greetings also fire
              the ``on_turn_start`` hook, so the bridge applies on
              the greeting turn too.
        inbox_merge_fn: Optional merge function for inbox payloads.
            Defaults to :func:`dict.update` (shallow merge).  Consumers
            supply deep-merge or conflict-resolving mergers as needed.
            Callable round-trip is by identity (in-process only — not
            JSON-serializable, mirroring other callable-bearing
            ``StructuredConfig`` fields).
    """

    wizard_config: str | dict[str, Any]
    config_base_path: str | None = None
    custom_functions: dict[str, Any] | None = None
    extraction_config: dict[str, Any] | None = None
    strict_validation: bool = True
    hooks: dict[str, Any] | None = None
    artifacts: dict[str, Any] | None = None
    review_protocols: dict[str, Any] | None = None
    initial_data: dict[str, Any] | None = None
    consistent_navigation_lifecycle: bool = True
    manager_metadata_inbox_key: str | list[str] | None = None
    inbox_merge_fn: Callable[[dict[str, Any], dict[str, Any]], None] | None = None

    def __post_init__(self) -> None:
        """Validate ``wizard_config`` is a path string or inline dict.

        The base ``from_dict`` projects non-coercible values verbatim, so a
        YAML scalar or list would otherwise pass through construction silently
        and only surface as a distant error when the wizard is loaded.  Guard
        it here, at the construction site, so the failure is local and clear.
        """
        if not isinstance(self.wizard_config, (str, dict)):
            raise ValueError(
                "wizard_config must be a path string or an inline "
                f"wizard-definition dict, got {type(self.wizard_config).__name__}"
            )
