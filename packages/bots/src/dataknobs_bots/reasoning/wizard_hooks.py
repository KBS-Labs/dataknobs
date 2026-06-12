"""Wizard lifecycle hooks for customizing wizard behavior.

This module provides WizardHooks for adding custom behavior at various
points in the wizard lifecycle, including stage transitions and completion.

Example:
    ```python
    from dataknobs_bots.reasoning.wizard_hooks import WizardHooks

    # Create hooks instance
    hooks = WizardHooks()

    # Add global hooks (apply to all stages)
    hooks.on_enter(lambda stage, data: print(f"Entering {stage}"))
    hooks.on_exit(lambda stage, data: print(f"Exiting {stage}"))

    # Add stage-specific hooks
    hooks.on_enter(my_welcome_handler, stage="welcome")

    # Add completion hook
    hooks.on_complete(lambda data: save_results(data))

    # Use with WizardReasoning
    reasoning = WizardReasoning(wizard_fsm=fsm, hooks=hooks)
    ```
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Union

from .function_resolver import resolve_function
from .lifecycle import LifecycleHooks, TurnHookCallback

logger = logging.getLogger(__name__)


# Type aliases for hook callbacks
StageCallback = Callable[[str, dict[str, Any]], Union[None, Awaitable[None]]]
CompleteCallback = Callable[[dict[str, Any]], Union[None, Awaitable[None]]]
ErrorCallback = Callable[
    [str, dict[str, Any], Exception], Union[None, Awaitable[None]]
]

# Re-export so existing consumers can ``from dataknobs_bots.reasoning.wizard_hooks
# import TurnHookCallback`` without learning the new module path.
__all__ = [
    "CompleteCallback",
    "ErrorCallback",
    "HookContext",
    "StageCallback",
    "TurnHookCallback",
    "WizardHooks",
]


@dataclass
class HookContext:
    """Context passed to hook callbacks.

    Provides additional information about the hook invocation
    beyond the basic stage name and data.

    Attributes:
        stage: Current stage name
        data: Wizard data dict
        previous_stage: Previous stage (for enter hooks)
        is_restart: True if this is a restart
        error: Exception if this is an error hook
    """

    stage: str
    data: dict[str, Any]
    previous_stage: str | None = None
    is_restart: bool = False
    error: Exception | None = None


@dataclass
class _HookRegistration:
    """Internal registration for a hook callback."""

    callback: Callable[..., Any]
    stage: str | None = None  # None means global (all stages)


class WizardHooks:
    """Lifecycle hooks for wizard stages and completion.

    WizardHooks provides a way to customize wizard behavior by
    registering callbacks for various lifecycle events:

    - **on_enter**: Called when entering a stage
    - **on_exit**: Called when leaving a stage
    - **on_complete**: Called when wizard completes
    - **on_restart**: Called when wizard is restarted
    - **on_error**: Called when an error occurs during processing

    Hooks can be registered globally (apply to all stages) or
    for specific stages only.

    Features:
        - Supports both sync and async callbacks
        - Stage-specific hooks for targeted customization
        - Error handling with dedicated error hooks
        - Configuration-based setup via from_config()

    Example:
        ```python
        hooks = WizardHooks()

        # Global enter hook
        async def log_enter(stage: str, data: dict) -> None:
            logger.info(f"Entering stage: {stage}")

        hooks.on_enter(log_enter)

        # Stage-specific hook
        def validate_email(stage: str, data: dict) -> None:
            if not data.get("email"):
                raise ValueError("Email required")

        hooks.on_exit(validate_email, stage="collect_email")

        # Completion hook
        hooks.on_complete(lambda data: save_to_database(data))
        ```

    Attributes:
        _enter_hooks: List of registered enter hooks
        _exit_hooks: List of registered exit hooks
        _complete_hooks: List of registered completion hooks
        _restart_hooks: List of registered restart hooks
        _error_hooks: List of registered error hooks
    """

    def __init__(self) -> None:
        """Initialize WizardHooks with empty hook registrations."""
        self._enter_hooks: list[_HookRegistration] = []
        self._exit_hooks: list[_HookRegistration] = []
        self._complete_hooks: list[CompleteCallback] = []
        self._restart_hooks: list[Callable[[], Any]] = []
        self._error_hooks: list[ErrorCallback] = []
        # Compose the strategy-agnostic turn-lifecycle surface. The
        # generic class owns registration / triggering / config-load
        # for the turn hooks; this wizard-specific class forwards
        # through it (preserving the wizard's error-handler fan-out).
        self._lifecycle: LifecycleHooks = LifecycleHooks()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WizardHooks":
        """Create WizardHooks from configuration dict.

        The configuration can specify hook module paths and functions
        to register. This allows declarative hook configuration in
        wizard YAML files.

        Args:
            config: Configuration dict with optional keys:
                - on_enter: List of {function: "module.path:func", stage: "name"}
                - on_exit: List of {function: "module.path:func", stage: "name"}
                - on_complete: List of "module.path:func" strings
                - on_restart: List of "module.path:func" strings
                - on_error: List of "module.path:func" strings

        Returns:
            Configured WizardHooks instance

        Example config:
            ```yaml
            hooks:
              on_enter:
                - function: "myapp.hooks:log_stage_entry"
                - function: "myapp.hooks:validate_welcome"
                  stage: welcome
              on_complete:
                - "myapp.hooks:finalize_wizard"
            ```
        """
        hooks = cls()

        # Register enter hooks
        for hook_config in config.get("on_enter", []):
            callback = cls._load_callback(hook_config)
            if callback:
                stage = (
                    hook_config.get("stage")
                    if isinstance(hook_config, dict)
                    else None
                )
                hooks.on_enter(callback, stage=stage)

        # Register exit hooks
        for hook_config in config.get("on_exit", []):
            callback = cls._load_callback(hook_config)
            if callback:
                stage = (
                    hook_config.get("stage")
                    if isinstance(hook_config, dict)
                    else None
                )
                hooks.on_exit(callback, stage=stage)

        # Register complete hooks
        for hook_ref in config.get("on_complete", []):
            callback = cls._load_callback(hook_ref)
            if callback:
                hooks.on_complete(callback)

        # Register restart hooks
        for hook_ref in config.get("on_restart", []):
            callback = cls._load_callback(hook_ref)
            if callback:
                hooks.on_restart(callback)

        # Register error hooks
        for hook_ref in config.get("on_error", []):
            callback = cls._load_callback(hook_ref)
            if callback:
                hooks.on_error(callback)

        # Delegate turn-lifecycle parsing to the generic LifecycleHooks
        # loader (same on_turn_start / on_turn_end keys + dotted-path
        # callbacks). Replace the embedded instance with the loaded one
        # rather than re-implementing parsing in two places.
        hooks._lifecycle = LifecycleHooks.from_config(config)

        return hooks

    @staticmethod
    def _load_callback(hook_config: dict[str, Any] | str) -> Callable[..., Any] | None:
        """Load a callback function from configuration.

        Args:
            hook_config: Either a string "module.path:function" or
                a dict with "function" key

        Returns:
            The loaded callback function, or None if loading failed
        """
        # Extract function reference
        if isinstance(hook_config, str):
            func_ref = hook_config
        elif isinstance(hook_config, dict):
            func_ref = hook_config.get("function", "")
        else:
            logger.warning("Invalid hook config type: %s", type(hook_config))
            return None

        if not func_ref:
            return None

        # Use shared function resolver with graceful error handling
        try:
            return resolve_function(func_ref)
        except (ValueError, ImportError, AttributeError) as e:
            logger.warning("Failed to load hook function '%s': %s", func_ref, e)
            return None

    def on_enter(
        self, callback: StageCallback, stage: str | None = None
    ) -> "WizardHooks":
        """Register a callback for stage entry.

        The callback is invoked when entering a stage, before
        any processing occurs.

        Args:
            callback: Function(stage_name, data) -> None or awaitable
            stage: Optional stage name to limit hook to

        Returns:
            Self for method chaining

        Example:
            ```python
            hooks.on_enter(lambda s, d: print(f"Entering {s}"))
            hooks.on_enter(validate_user, stage="user_info")
            ```
        """
        self._enter_hooks.append(_HookRegistration(callback=callback, stage=stage))
        return self

    def on_exit(
        self, callback: StageCallback, stage: str | None = None
    ) -> "WizardHooks":
        """Register a callback for stage exit.

        The callback is invoked when leaving a stage, after
        data collection but before the transition.

        Args:
            callback: Function(stage_name, data) -> None or awaitable
            stage: Optional stage name to limit hook to

        Returns:
            Self for method chaining

        Example:
            ```python
            hooks.on_exit(lambda s, d: audit_log(s, d))
            hooks.on_exit(validate_config, stage="configure")
            ```
        """
        self._exit_hooks.append(_HookRegistration(callback=callback, stage=stage))
        return self

    def on_complete(self, callback: CompleteCallback) -> "WizardHooks":
        """Register a callback for wizard completion.

        The callback is invoked when the wizard reaches
        its end state.

        Args:
            callback: Function(data) -> None or awaitable

        Returns:
            Self for method chaining

        Example:
            ```python
            hooks.on_complete(lambda d: save_results(d))
            hooks.on_complete(send_confirmation_email)
            ```
        """
        self._complete_hooks.append(callback)
        return self

    def on_restart(self, callback: Callable[[], Any]) -> "WizardHooks":
        """Register a callback for wizard restart.

        The callback is invoked when the wizard is restarted,
        before resetting to the initial stage.

        Args:
            callback: Function() -> None or awaitable

        Returns:
            Self for method chaining

        Example:
            ```python
            hooks.on_restart(lambda: clear_temp_files())
            ```
        """
        self._restart_hooks.append(callback)
        return self

    def on_error(self, callback: ErrorCallback) -> "WizardHooks":
        """Register a callback for error handling.

        The callback is invoked when an error occurs during
        wizard processing.

        Args:
            callback: Function(stage_name, data, error) -> None or awaitable

        Returns:
            Self for method chaining

        Example:
            ```python
            hooks.on_error(lambda s, d, e: log_error(s, e))
            ```
        """
        self._error_hooks.append(callback)
        return self

    # ── Turn-lifecycle pass-through ──────────────────────────────
    # Forwarded to the embedded LifecycleHooks so registration looks
    # the same to consumers (``hooks.on_turn_start(cb)``). Triggers
    # wrap the LifecycleHooks call in the wizard's existing error-
    # handler fan-out so a failing hook still reaches ``on_error``
    # callbacks.

    def on_turn_start(
        self,
        callback: TurnHookCallback,
        *,
        stage: str | None = None,
    ) -> "WizardHooks":
        """Register a callback for the start of every wizard turn.

        Fires from :meth:`WizardReasoning.begin_turn` AFTER the
        per-turn ephemeral-key clear, BEFORE the user-message read
        and any early-return dispatch. Also fires from
        :meth:`WizardReasoning.greet` so bot-initiated greetings
        inherit the surface.

        Common consumer uses: bridging sub-strategy signals into
        transition-eval scope, tenant context binding, audit trail
        injection, custom validation gating.

        Args:
            callback: ``(event: dict[str, Any])`` — sync or async.
                See :mod:`dataknobs_bots.reasoning.lifecycle` for
                the canonical event-payload keys
                (``stage`` / ``phase`` / ``reason`` / ``manager`` /
                ``state``). The wizard publishes ``reason="normal"``
                from the canonical turn-start fire-points.
            stage: Optional stage name to limit the hook to.

        Returns:
            Self, for chaining.
        """
        self._lifecycle.on_turn_start(callback, stage=stage)
        return self

    def on_turn_end(
        self,
        callback: TurnHookCallback,
        *,
        stage: str | None = None,
    ) -> "WizardHooks":
        """Register a callback for the end of every wizard turn.

        Fires from :meth:`WizardReasoning.finalize_turn` after the
        state-save round-trip and from every early-return exit
        (amendment / navigation / validation / collection-mode /
        confirmation / clarification) with the matching ``reason``
        key. Also fires from ``stream_finalize_turn`` (including
        the ``aclose()`` abandonment path with ``reason="abandoned"``)
        and from the non-conversational :meth:`advance` API with
        ``reason="advance"``. See :meth:`on_turn_start` for the
        callback contract.
        """
        self._lifecycle.on_turn_end(callback, stage=stage)
        return self

    async def trigger_turn_start(self, event: dict[str, Any]) -> None:
        """Trigger all registered ``on_turn_start`` hooks.

        Wraps the LifecycleHooks call in the wizard's existing
        error-handler fan-out (``_handle_error``) so a failing hook
        still reaches ``on_error`` callbacks before re-raising.
        """
        try:
            await self._lifecycle.trigger_turn_start(event)
        except Exception as e:
            await self._handle_error_from_event(event, e)
            raise

    async def trigger_turn_end(self, event: dict[str, Any]) -> None:
        """Trigger all registered ``on_turn_end`` hooks. See
        :meth:`trigger_turn_start`.
        """
        try:
            await self._lifecycle.trigger_turn_end(event)
        except Exception as e:
            await self._handle_error_from_event(event, e)
            raise

    async def _handle_error_from_event(
        self, event: dict[str, Any], error: Exception,
    ) -> None:
        """Bridge an event-shaped failure back into the legacy
        ``on_error`` fan-out, which expects ``(stage, data, error)``.
        """
        stage = event.get("stage", "")
        state = event.get("state")
        data = getattr(state, "data", {}) if state is not None else {}
        if not isinstance(data, dict):
            data = {}
        await self._handle_error(stage, data, error)

    @property
    def lifecycle(self) -> LifecycleHooks:
        """Read-only access to the embedded :class:`LifecycleHooks`.

        Useful when a consumer wants to detach the lifecycle surface
        (e.g. install the wizard's lifecycle hooks on a non-wizard
        sibling strategy in the same conversation).
        """
        return self._lifecycle

    async def trigger_enter(self, stage: str, data: dict[str, Any]) -> None:
        """Trigger all registered enter hooks for a stage.

        Invokes global hooks and stage-specific hooks that match
        the given stage name.

        Args:
            stage: Name of the stage being entered
            data: Current wizard data

        Raises:
            Exception: Re-raises any exception from hooks after
                triggering error hooks
        """
        for registration in self._enter_hooks:
            if registration.stage is None or registration.stage == stage:
                try:
                    await self._invoke_callback(registration.callback, stage, data)
                except Exception as e:
                    await self._handle_error(stage, data, e)
                    raise

    async def trigger_exit(self, stage: str, data: dict[str, Any]) -> None:
        """Trigger all registered exit hooks for a stage.

        Invokes global hooks and stage-specific hooks that match
        the given stage name.

        Args:
            stage: Name of the stage being exited
            data: Current wizard data

        Raises:
            Exception: Re-raises any exception from hooks after
                triggering error hooks
        """
        for registration in self._exit_hooks:
            if registration.stage is None or registration.stage == stage:
                try:
                    await self._invoke_callback(registration.callback, stage, data)
                except Exception as e:
                    await self._handle_error(stage, data, e)
                    raise

    async def trigger_complete(self, data: dict[str, Any]) -> None:
        """Trigger all registered completion hooks.

        Invokes all completion hooks with the final wizard data.

        Args:
            data: Final wizard data

        Raises:
            Exception: Re-raises any exception from hooks after
                triggering error hooks
        """
        for callback in self._complete_hooks:
            try:
                await self._invoke_callback(callback, data)
            except Exception as e:
                await self._handle_error("_complete", data, e)
                raise

    async def trigger_restart(self) -> None:
        """Trigger all registered restart hooks.

        Invokes all restart hooks before wizard reset.

        Raises:
            Exception: Re-raises any exception from hooks after
                triggering error hooks
        """
        for callback in self._restart_hooks:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                await self._handle_error("_restart", {}, e)
                raise

    async def _invoke_callback(
        self, callback: Callable[..., Any], *args: Any
    ) -> None:
        """Invoke a callback, handling both sync and async.

        Args:
            callback: The callback to invoke
            *args: Arguments to pass to the callback
        """
        result = callback(*args)
        if asyncio.iscoroutine(result):
            await result

    async def _handle_error(
        self, stage: str, data: dict[str, Any], error: Exception
    ) -> None:
        """Invoke error hooks for an exception.

        Args:
            stage: Stage where error occurred
            data: Wizard data at time of error
            error: The exception that occurred
        """
        logger.error("Hook error in stage %s: %s", stage, error)

        for callback in self._error_hooks:
            try:
                result = callback(stage, data, error)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                # Don't let error handlers cause more errors
                logger.exception("Error in error handler")

    def clear(self) -> None:
        """Clear ALL registered hooks, including the embedded
        :class:`LifecycleHooks` turn surface.

        Reset semantics cover every hook type a consumer can register
        through this class — enter / exit / complete / restart / error
        (legacy hooks) and turn_start / turn_end (delegated to the
        composed lifecycle instance). The lifecycle instance identity
        is preserved (drained in place), so any external reference held
        via the :attr:`lifecycle` property remains valid post-clear.
        """
        self._enter_hooks.clear()
        self._exit_hooks.clear()
        self._complete_hooks.clear()
        self._restart_hooks.clear()
        self._error_hooks.clear()
        self._lifecycle.clear()

    @property
    def hook_count(self) -> dict[str, int]:
        """Get count of registered hooks by type.

        Returns:
            Dict mapping hook type to count. Includes the composed
            turn-lifecycle surface (``turn_start`` / ``turn_end``)
            alongside the legacy wizard hooks.
        """
        return {
            "enter": len(self._enter_hooks),
            "exit": len(self._exit_hooks),
            "complete": len(self._complete_hooks),
            "restart": len(self._restart_hooks),
            "error": len(self._error_hooks),
            "turn_start": self._lifecycle.turn_start_count,
            "turn_end": self._lifecycle.turn_end_count,
        }
