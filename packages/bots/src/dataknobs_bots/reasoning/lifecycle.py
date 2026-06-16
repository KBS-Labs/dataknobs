"""Strategy-agnostic turn-lifecycle hook surface.

:class:`LifecycleHooks` is the importable extension surface for any
:class:`~dataknobs_bots.reasoning.base.ReasoningStrategy`
implementation that wants pre-turn / post-turn extension. The
wizard's :class:`WizardHooks` composes this class; consumer
composing strategies (e.g. pipeline-shaped strategies) adopt by:

1. Accepting an optional ``hooks: LifecycleHooks | None = None``
   constructor parameter.
2. Building an event dict and firing
   ``await hooks.trigger_turn_start(event)`` at the start of each
   turn (after per-turn ephemeral state is reset; before any
   early-return dispatch).
3. Building an event dict and firing
   ``await hooks.trigger_turn_end(event)`` at the end of each turn,
   including early-return / abandonment exits — discriminate exit
   type via the documented ``reason`` key on the event.

Hooks are loadable from config via :meth:`from_config` with
dotted-path callback resolution — same shape consumers already
know from :class:`WizardHooks`.

Per-strategy / per-stage scoping is the responsibility of each
adopting strategy: stage-scoped registrations wrap the callback
with a filter on ``event["stage"]`` (the wizard passes the current
FSM state name; a pipeline strategy could pass the active step or
a constant ``"pipeline"``).

Event payload
-------------

Triggers carry a single ``event: dict[str, Any]`` argument. The
contract is intentionally narrow at the protocol level (it's an
opaque dict so adopters add subsystem-specific keys without
extending the trigger signature), but adopting strategies are
expected to populate the following common keys:

- ``stage``: ``str`` — the stage / step name the trigger fires
  against. The trigger's stage-scope matching reads this key.
- ``phase``: ``"start"`` or ``"end"`` — discriminator for callbacks
  that registered against both surfaces.
- ``reason``: ``str`` — discriminator for *why* the turn is
  entering / exiting. The wizard publishes ``"normal"`` for the
  canonical finalize-turn exit, ``"amendment"`` / ``"navigation"``
  / ``"clarification"`` / ``"collection_help"`` /
  ``"collection_loop"`` / ``"confirmation"`` / ``"validation_error"``
  for early-return exits, ``"abandoned"`` for stream abandonment,
  and ``"advance"`` for the non-conversational ``advance()`` API.
- ``manager``: the conversation manager (or ``None`` when the
  trigger fires from a non-conversational API like the wizard's
  ``advance()``).
- ``state``: the strategy-specific state object (e.g.
  ``WizardState`` for the wizard).

Today's surface is intentionally narrow (just turn_start /
turn_end). Additional lifecycle points (on_transition,
on_response_generated, on_step_start, etc.) are captured as
follow-up rather than speculated here — when the next adopter
surfaces a concrete need, the right shape will be clear.

Substrate
---------

Dispatch is delegated to a
:class:`~dataknobs_common.callbacks.CallbackRegistry` (one registry
per :class:`LifecycleHooks` instance, two topics:
``turn_start`` and ``turn_end``). Consumer-facing methods are
unchanged from the pre-substrate implementation; the registry is
exposed read-only via the :attr:`registry` property so consumers
wanting pluggable ordering (e.g. :class:`PriorityOrdering`),
priority-tagged callbacks, per-registry error policies, or
:meth:`~dataknobs_common.callbacks.CallbackRegistry.also_publish_to`
EventBus fan-out can reach through without monkey-patching. The
:class:`~dataknobs_common.capabilities.Capability.CALLBACK_REGISTRY`
capability advertises this composition path.

**Proposed topic-naming convention** for cross-strategy bus
subscriptions: ``<subsystem>:<operation>:<phase>`` — e.g.
``wizard:turn:start``, ``pipeline:turn:end``,
``ingest:domain:start``. The wizard adopter wires this through
``hooks.registry.also_publish_to(bus, topic_prefix="wizard:")``;
the convention is forward-compatible across every adopting
strategy without protocol-level coupling.
"""
from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, Union

from dataknobs_common.callbacks import (
    CallbackRegistry,
    ErrorPolicy,
    FIFOOrdering,
)
from dataknobs_common.capabilities import Capability, CapabilityMixin

logger = logging.getLogger(__name__)


# Callback signature for turn-lifecycle hooks. The single argument is
# the opaque event dict described in the module docstring — adopters
# add subsystem-specific keys without extending the trigger signature.
#
# Sync OR async return is accepted (the registry awaits awaitable
# returns and treats non-awaitable returns as already-complete).
TurnHookCallback = Callable[[dict[str, Any]], Union[Awaitable[None], None]]


class LifecycleHooks(CapabilityMixin):
    """Strategy-agnostic turn-lifecycle hook registry.

    Construct standalone or compose into a strategy-specific hook
    class (the wizard's :class:`WizardHooks` does the latter).

    Dispatch is backed by a
    :class:`~dataknobs_common.callbacks.CallbackRegistry`. The
    consumer-facing surface (:meth:`on_turn_start` /
    :meth:`on_turn_end` registration, :meth:`trigger_turn_start` /
    :meth:`trigger_turn_end` triggers, :meth:`clear`,
    :meth:`from_config`) is byte-identical to the pre-substrate
    implementation; the :attr:`registry` accessor exposes the
    underlying registry for consumers wanting pluggable ordering,
    priority-tagged callbacks, or EventBus fan-out.
    """

    SUPPORTED_CAPABILITIES: ClassVar[frozenset[Capability]] = frozenset({
        Capability.CALLBACK_REGISTRY,
    })

    _TOPIC_TURN_START = "turn_start"
    _TOPIC_TURN_END = "turn_end"

    def __init__(self) -> None:
        # FIFO ordering + LOG_AND_CONTINUE policy match the
        # pre-substrate behavior. Consumers wanting priority /
        # staged dispatch reach through ``self.registry`` and call
        # ``set_ordering(...)`` before any registration.
        self._registry: CallbackRegistry[TurnHookCallback] = CallbackRegistry(
            ordering=FIFOOrdering(),
            error_policy=ErrorPolicy.LOG_AND_CONTINUE,
        )

    # ── Registration ──────────────────────────────────────────────

    def on_turn_start(
        self,
        callback: TurnHookCallback,
        *,
        stage: str | None = None,
    ) -> LifecycleHooks:
        """Register a turn-start callback.

        Args:
            callback: Function ``(event: dict[str, Any])`` — sync or
                async. See the module docstring for the canonical
                event-payload keys (``stage`` / ``phase`` / ``reason`` /
                ``manager`` / ``state``).
            stage: Optional stage name to limit the hook to. ``None``
                (default) fires for every stage. Matched against
                ``event["stage"]``.

        Returns:
            Self, for chaining.
        """
        wrapped = self._stage_scoped(callback, stage) if stage else callback
        self._registry.register(self._TOPIC_TURN_START, wrapped)
        return self

    def on_turn_end(
        self,
        callback: TurnHookCallback,
        *,
        stage: str | None = None,
    ) -> LifecycleHooks:
        """Register a turn-end callback. See :meth:`on_turn_start`."""
        wrapped = self._stage_scoped(callback, stage) if stage else callback
        self._registry.register(self._TOPIC_TURN_END, wrapped)
        return self

    # ── Triggering ────────────────────────────────────────────────

    async def trigger_turn_start(self, event: dict[str, Any]) -> None:
        """Fire all matching turn-start callbacks (global + this stage).

        Stage-scoped registrations match when ``event["stage"]`` equals
        the registered stage; global registrations (``stage=None``)
        always fire. Callbacks fire in registration order under the
        default :class:`~dataknobs_common.callbacks.FIFOOrdering`; a
        consumer can swap orderings via
        ``hooks.registry.set_ordering(...)``. Sync callbacks return
        immediately; async callbacks are awaited before the next runs.
        """
        await self._registry.fire_async(self._TOPIC_TURN_START, event)

    async def trigger_turn_end(self, event: dict[str, Any]) -> None:
        """Fire all matching turn-end callbacks. See :meth:`trigger_turn_start`."""
        await self._registry.fire_async(self._TOPIC_TURN_END, event)

    # ── Public introspection / reset ──────────────────────────────

    @property
    def registry(self) -> CallbackRegistry[TurnHookCallback]:
        """Read-only accessor exposing the underlying
        :class:`~dataknobs_common.callbacks.CallbackRegistry`.

        Use this when a consumer wants to plug in a custom ordering,
        register a priority-tagged callback, or fan out lifecycle
        events to an :class:`~dataknobs_common.events.EventBus`::

            hooks = LifecycleHooks()
            hooks.registry.set_ordering(PriorityOrdering())
            hooks.registry.register(
                "turn_start", emergency_guard, priority=-100,
            )
            hooks.registry.also_publish_to(bus, topic_prefix="wizard:")

        The registry instance survives :meth:`clear` (which drains
        in place), so a reference held here stays valid across resets.
        """
        return self._registry

    @property
    def turn_start_count(self) -> int:
        """Number of registered ``on_turn_start`` callbacks."""
        return self._registry.callback_count(self._TOPIC_TURN_START)

    @property
    def turn_end_count(self) -> int:
        """Number of registered ``on_turn_end`` callbacks."""
        return self._registry.callback_count(self._TOPIC_TURN_END)

    def clear(self) -> None:
        """Clear all registered turn-lifecycle callbacks.

        Drains the underlying registry in place — the registry
        instance identity is preserved, so consumers holding a
        reference via :attr:`registry` (or via
        :attr:`WizardHooks.lifecycle`) remain attached and can
        re-register without rewiring.
        """
        self._registry.clear()

    # ── Stage-scope wrapping ──────────────────────────────────────

    @staticmethod
    def _stage_scoped(
        callback: TurnHookCallback,
        stage: str,
    ) -> TurnHookCallback:
        """Wrap ``callback`` so it only fires when ``event['stage']``
        equals the registered stage.

        Async-shape-preserving: an async callback is wrapped in an
        async wrapper (so the registry's
        :meth:`~dataknobs_common.callbacks.CallbackRegistry.fire_async`
        awaits the inner result correctly); a sync callback gets a
        sync wrapper.
        """
        if inspect.iscoroutinefunction(callback):
            async def async_wrapped(event: dict[str, Any]) -> None:
                if event.get("stage") == stage:
                    await callback(event)  # type: ignore[misc]
            return async_wrapped

        def sync_wrapped(event: dict[str, Any]) -> None:
            if event.get("stage") == stage:
                callback(event)
        return sync_wrapped

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
