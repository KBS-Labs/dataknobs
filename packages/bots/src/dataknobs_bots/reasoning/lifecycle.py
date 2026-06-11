"""Strategy-agnostic turn-lifecycle hook surface.

:class:`LifecycleHooks` is the importable extension surface for any
:class:`~dataknobs_bots.reasoning.base.ReasoningStrategy`
implementation that wants pre-turn / post-turn extension. The
wizard's :class:`WizardHooks` composes this class; consumer
composing strategies (e.g. pipeline-shaped strategies) adopt by:

1. Accepting an optional ``hooks: LifecycleHooks | None = None``
   constructor parameter.
2. Firing ``await hooks.trigger_turn_start(...)`` at the start of
   each turn (after per-turn ephemeral state is reset; before
   any early-return dispatch).
3. Firing ``await hooks.trigger_turn_end(...)`` at the end of each
   turn (after the strategy's state-save / response-return).

Hooks are loadable from config via :meth:`from_config` with
dotted-path callback resolution — same shape consumers already
know from :class:`WizardHooks`.

Per-strategy / per-stage scoping is the responsibility of each
adopting strategy: pass a meaningful ``stage_name`` string to
``trigger_turn_start`` / ``trigger_turn_end`` (e.g. the wizard
passes the current FSM state name; a pipeline strategy could pass
the active step or a constant ``"pipeline"``).

Today's surface is intentionally narrow (just turn_start /
turn_end). Additional lifecycle points (on_transition,
on_response_generated, on_step_start, etc.) are captured as
follow-up rather than speculated here — when the next adopter
surfaces a concrete need, the right shape will be clear.

**Lift trajectory (follow-up).** This class is one of seven
in-process named-callback registries currently inline in
``dataknobs-bots`` (the wizard's ``on_enter`` / ``on_exit`` /
``on_complete`` / ``on_restart`` / ``on_error`` + this class's
``on_turn_start`` / ``on_turn_end`` + ``ToolRegistry``'s
``ExecutionTracker``). A future consolidation can extract the
shared registration-trigger-from_config pattern into a generic
``CallbackRegistry[CallbackT]`` in ``dataknobs_common.callbacks``;
each of the seven adopters becomes a thin typed wrapper. The same
consolidation can layer an optional ``EventBus`` substrate so
external pub-sub systems (observability, alerting, webhook
delivery, cross-process audit) attach as event subscribers without
writing a custom hook per system. In-process callback semantics
would be unchanged across both layers (this class's typed
``on_turn_start`` / ``on_turn_end`` API stays as-is); external
fan-out would be opt-in via a future ``also_publish_to(bus,
topic_prefix=...)`` method.

**Proposed topic-naming convention** for the future bus layer:
``lifecycle:<strategy>:<event>:<scope?>`` — e.g.
``lifecycle:wizard:turn:start:propose``,
``lifecycle:pipeline:turn:end``. Seeded here so when the bus layer
ships, the cross-strategy subscription convention is already set.
Consumer adoption of the typed surface today is forward-compatible
with both the in-process consolidation and the opt-in bus
substrate.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Union

logger = logging.getLogger(__name__)


# Callback signature for turn-lifecycle hooks. Matches the existing
# WizardHooks shape: (manager, state, stage_name).
#
# Sync OR async return is accepted (the trigger awaits awaitable
# returns and treats non-awaitable returns as already-complete).
TurnHookCallback = Callable[[Any, Any, str], Union[Awaitable[None], None]]


@dataclass(frozen=True)
class _HookRegistration:
    """One registered hook + optional stage scope."""

    callback: TurnHookCallback
    stage: str | None = None  # None = global; non-None = stage-scoped


class LifecycleHooks:
    """Strategy-agnostic turn-lifecycle hook registry.

    Construct standalone or compose into a strategy-specific hook
    class (the wizard's :class:`WizardHooks` does the latter).
    """

    def __init__(self) -> None:
        self._turn_start_hooks: list[_HookRegistration] = []
        self._turn_end_hooks: list[_HookRegistration] = []

    # ── Registration ──────────────────────────────────────────────

    def on_turn_start(
        self,
        callback: TurnHookCallback,
        *,
        stage: str | None = None,
    ) -> LifecycleHooks:
        """Register a turn-start callback.

        Args:
            callback: Function ``(manager, state, stage_name)`` — sync
                or async.
            stage: Optional stage name to limit the hook to. ``None``
                (default) fires for every stage.

        Returns:
            Self, for chaining.
        """
        self._turn_start_hooks.append(_HookRegistration(callback, stage))
        return self

    def on_turn_end(
        self,
        callback: TurnHookCallback,
        *,
        stage: str | None = None,
    ) -> LifecycleHooks:
        """Register a turn-end callback. See :meth:`on_turn_start`."""
        self._turn_end_hooks.append(_HookRegistration(callback, stage))
        return self

    # ── Triggering ────────────────────────────────────────────────

    async def trigger_turn_start(
        self, manager: Any, state: Any, stage_name: str,
    ) -> None:
        """Fire all matching turn-start callbacks (global + this stage).

        Callbacks fire in registration order. Sync callbacks return
        immediately; async callbacks are awaited before the next runs.
        """
        for reg in self._turn_start_hooks:
            if reg.stage is None or reg.stage == stage_name:
                await self._invoke(reg.callback, manager, state, stage_name)

    async def trigger_turn_end(
        self, manager: Any, state: Any, stage_name: str,
    ) -> None:
        """Fire all matching turn-end callbacks. See :meth:`trigger_turn_start`."""
        for reg in self._turn_end_hooks:
            if reg.stage is None or reg.stage == stage_name:
                await self._invoke(reg.callback, manager, state, stage_name)

    @staticmethod
    async def _invoke(
        callback: TurnHookCallback, manager: Any, state: Any, stage_name: str,
    ) -> None:
        result = callback(manager, state, stage_name)
        if asyncio.iscoroutine(result):
            await result

    # ── Public introspection / reset ──────────────────────────────

    @property
    def turn_start_count(self) -> int:
        """Number of registered ``on_turn_start`` callbacks."""
        return len(self._turn_start_hooks)

    @property
    def turn_end_count(self) -> int:
        """Number of registered ``on_turn_end`` callbacks."""
        return len(self._turn_end_hooks)

    def clear(self) -> None:
        """Clear all registered turn-lifecycle callbacks."""
        self._turn_start_hooks.clear()
        self._turn_end_hooks.clear()

    # ── Config-driven loading ─────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LifecycleHooks:
        """Build a :class:`LifecycleHooks` from a config dict.

        Accepts the same dotted-path callback shape as the wizard's
        :class:`WizardHooks` loader:

        .. code-block:: yaml

           on_turn_start:
             - function: "myapp.hooks:bind_tenant"
           on_turn_end:
             - function: "myapp.hooks:emit_audit"
               stage: "triage"

        A list entry may also be a bare ``"module.path:name"`` string
        when no stage scoping is needed.
        """
        hooks = cls()
        for entry in config.get("on_turn_start", []):
            callback, stage = cls._resolve_entry(entry)
            if callback is not None:
                hooks.on_turn_start(callback, stage=stage)
        for entry in config.get("on_turn_end", []):
            callback, stage = cls._resolve_entry(entry)
            if callback is not None:
                hooks.on_turn_end(callback, stage=stage)
        return hooks

    @staticmethod
    def _resolve_entry(
        entry: dict[str, Any] | str,
    ) -> tuple[TurnHookCallback | None, str | None]:
        if isinstance(entry, str):
            path: str | None = entry
            stage: str | None = None
        elif isinstance(entry, dict):
            path = entry.get("function") or entry.get("callback")
            stage = entry.get("stage")
        else:
            logger.warning("Invalid lifecycle hook entry type: %s", type(entry))
            return None, None

        if not path:
            return None, None

        module_path, _, name = path.partition(":")
        if not module_path or not name:
            logger.warning(
                "Lifecycle hook callback path must be 'module.path:name'; got %r",
                path,
            )
            return None, None

        try:
            module = importlib.import_module(module_path)
            return getattr(module, name), stage
        except (ImportError, AttributeError) as exc:
            logger.warning(
                "Failed to resolve lifecycle hook callback %r: %s", path, exc,
            )
            return None, None
