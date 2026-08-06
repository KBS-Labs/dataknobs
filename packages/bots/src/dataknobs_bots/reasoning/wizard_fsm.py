"""WizardFSM wrapper for wizard-specific FSM operations.

This module provides a thin wrapper around AdvancedFSM that adds
wizard-specific conveniences like stage metadata access, navigation
helpers, and serialization for persistence.
"""

import copy
import logging
from types import TracebackType
from typing import Any, Callable, Self

from dataknobs_fsm.api.advanced import AdvancedFSM, StepResult
from dataknobs_fsm.execution.context import ExecutionContext

logger = logging.getLogger(__name__)


class WizardFSM:
    """Wrapper around AdvancedFSM with wizard-specific conveniences.

    Provides a simplified interface for wizard operations including:
    - Stage metadata access (prompts, schemas, suggestions)
    - Navigation helpers (back, skip, restart)
    - State serialization for persistence
    - Stage-specific tool and configuration access
    - Subflow registry for nested wizard flows

    Lifecycle: the wrapped ``AdvancedFSM`` allocates a daemon event-loop
    thread the first time it is stepped synchronously, so a wizard driven
    through :meth:`step` holds an OS resource. Release it with
    :meth:`close` / :meth:`aclose`, or let ``with`` / ``async with`` do it
    by construction. Closing cascades to every subflow this FSM owns.

    Attributes:
        _fsm: Underlying AdvancedFSM instance
        _stage_metadata: Dict mapping stage names to metadata
        _settings: Wizard-level settings (auto_advance_filled_stages, etc.)
        _context: Current execution context
        _subflow_registry: Dict mapping subflow names to WizardFSM instances
        _owns_subflows: Names in ``_subflow_registry`` whose lifecycle this
            FSM owns, and which its close therefore cascades to
    """

    def __init__(
        self,
        fsm: AdvancedFSM,
        stage_metadata: dict[str, dict[str, Any]],
        settings: dict[str, Any] | None = None,
        subflow_registry: dict[str, "WizardFSM"] | None = None,
        transform_context_factory: Callable[..., Any] | None = None,
    ):
        """Initialize WizardFSM.

        Args:
            fsm: AdvancedFSM instance to wrap
            stage_metadata: Dict mapping stage names to their metadata
            settings: Wizard-level settings dict (optional)
            subflow_registry: Dict mapping subflow names to WizardFSM
                instances. Subflows passed here are **owned** by this FSM —
                they are built for it by the loader, so :meth:`close`
                cascades to them. Use :meth:`register_subflow` with
                ``owns=False`` to add one whose lifecycle stays with its
                caller.
            transform_context_factory: Optional callable that receives a
                :class:`FunctionContext` and returns the application-specific
                context for transforms (e.g. :class:`TransformContext`).
                Can also be set later via
                :meth:`set_transform_context_factory`.
        """
        self._fsm = fsm
        self._stage_metadata = stage_metadata
        self._settings = settings or {}
        self._context: ExecutionContext | None = None
        self._subflow_registry: dict[str, WizardFSM] = subflow_registry or {}
        self._owns_subflows: set[str] = set(self._subflow_registry)
        self._transform_context_factory: Callable[..., Any] | None = (
            transform_context_factory
        )

    def set_transform_context_factory(
        self, factory: Callable[..., Any]
    ) -> None:
        """Register a factory for building transform-level context objects.

        The factory receives a :class:`FunctionContext` and returns the
        application-specific context that transforms should receive (e.g.
        :class:`TransformContext`).  It is applied to the
        :class:`ExecutionContext` before each step executes.

        If the :class:`ExecutionContext` has already been created (i.e.
        after the first step), the factory is propagated to it immediately
        so that the next ``step`` / ``step_async`` call uses the new
        factory without requiring context re-creation.

        Args:
            factory: Callable accepting a FunctionContext and returning
                the desired transform context.
        """
        self._transform_context_factory = factory
        if self._context is not None:
            self._context.transform_context_factory = factory

    def get_transform_context_factory(self) -> Callable[..., Any] | None:
        """Return the currently registered transform context factory.

        Returns:
            The factory callable, or ``None`` if none is registered.
        """
        return self._transform_context_factory

    @property
    def settings(self) -> dict[str, Any]:
        """Get wizard-level settings.

        Returns:
            Dict containing wizard settings like auto_advance_filled_stages
        """
        return self._settings

    @property
    def current_stage(self) -> str:
        """Get current stage name.

        Returns:
            Name of the current stage
        """
        if self._context and self._context.current_state:
            return self._context.current_state
        return self._find_start_stage()

    @property
    def current_metadata(self) -> dict[str, Any]:
        """Get metadata for current stage.

        Returns:
            Dict containing current stage's metadata
        """
        return self._stage_metadata.get(self.current_stage, {})

    @property
    def stages(self) -> dict[str, dict[str, Any]]:
        """Get all stage metadata.

        Returns a copy to prevent external modification.

        Returns:
            Dict mapping stage name to stage configuration dict.
        """
        return dict(self._stage_metadata)

    @property
    def stage_names(self) -> list[str]:
        """Get ordered list of stage names.

        Returns:
            List of stage names in definition order.
        """
        return list(self._stage_metadata.keys())

    @property
    def stage_count(self) -> int:
        """Get total number of stages.

        Returns:
            Number of stages in the wizard.
        """
        return len(self._stage_metadata)

    def get_stage_prompt(self, stage: str | None = None) -> str:
        """Get prompt for a stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            Stage prompt string
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("prompt", "")

    def get_stage_schema(self, stage: str | None = None) -> dict[str, Any] | None:
        """Get validation schema for a stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            JSON Schema dict or None
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("schema")

    def get_stage_tools(self, stage: str | None = None) -> list[str]:
        """Get available tool names for a stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            List of tool names available in the stage
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("tools", [])

    def get_stage_suggestions(self, stage: str | None = None) -> list[str]:
        """Get quick-reply suggestions for a stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            List of suggestion strings
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("suggestions", [])

    def get_transition_condition(
        self, from_stage: str, to_stage: str
    ) -> str | None:
        """Get the condition expression for a transition.

        Args:
            from_stage: Source stage name
            to_stage: Target stage name

        Returns:
            Condition expression string, or None if no condition
        """
        stage_meta = self._stage_metadata.get(from_stage, {})
        transitions = stage_meta.get("transitions", [])

        for transition in transitions:
            if transition.get("target") == to_stage:
                return transition.get("condition")

        return None

    def can_skip(self, stage: str | None = None) -> bool:
        """Check if stage can be skipped.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            True if stage can be skipped
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("can_skip", False)

    def can_go_back(self, stage: str | None = None) -> bool:
        """Check if back navigation is allowed.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            True if back navigation is allowed
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("can_go_back", True)

    def is_start_stage(self, stage: str | None = None) -> bool:
        """Check if stage is a start stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            True if stage is marked as start
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("is_start", False)

    def is_end_stage(self, stage: str | None = None) -> bool:
        """Check if stage is an end stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            True if stage is marked as end
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("is_end", False)

    # =========================================================================
    # Subflow Registry Methods
    # =========================================================================

    def get_subflow(self, name: str) -> "WizardFSM | None":
        """Get a subflow WizardFSM by name.

        Args:
            name: Subflow network name

        Returns:
            WizardFSM instance for the subflow, or None if not found
        """
        return self._subflow_registry.get(name)

    def has_subflow(self, name: str) -> bool:
        """Check if a subflow exists in the registry.

        Args:
            name: Subflow network name

        Returns:
            True if the subflow exists
        """
        return name in self._subflow_registry

    def register_subflow(
        self, name: str, subflow_fsm: "WizardFSM", *, owns: bool = True
    ) -> None:
        """Register a subflow WizardFSM.

        Args:
            name: Subflow network name
            subflow_fsm: WizardFSM instance for the subflow
            owns: Whether this FSM owns *subflow_fsm*'s lifecycle. When
                True (the default, matching the loader-built subflows this
                registry normally holds), :meth:`close` and :meth:`aclose`
                cascade to it. Pass False to register a subflow the caller
                built and still holds — closing it out from under its owner
                would tear down an FSM that may still be stepped.

                Re-registering a name replaces both the subflow and its
                ownership, so a name is owned iff its *most recent*
                registration said so.
        """
        self._subflow_registry[name] = subflow_fsm
        if owns:
            self._owns_subflows.add(name)
        else:
            self._owns_subflows.discard(name)

    @property
    def subflow_names(self) -> list[str]:
        """Get list of registered subflow names.

        Returns:
            List of subflow network names
        """
        return list(self._subflow_registry.keys())

    def resolve_function(self, name: str) -> Callable[..., Any] | None:
        """Look up a registered function by name.

        Searches the FSM's function registry (functions, validators,
        and transforms) and custom functions for a callable matching
        *name*.  Used by :class:`WizardReasoning` to resolve routing
        transform names declared in stage config.

        Args:
            name: Function name as registered in the FSM.

        Returns:
            The callable, or ``None`` if not found.
        """
        registry = getattr(self._fsm.fsm, "function_registry", None)
        if registry is not None and hasattr(registry, "get_function"):
            return registry.get_function(name)
        return None

    def step(self, data: dict[str, Any]) -> StepResult:
        """Execute one FSM step with given data.

        Creates or updates the execution context and executes
        a single FSM transition.

        Args:
            data: Data dict for transition evaluation

        Returns:
            StepResult with transition details
        """
        before_stage = self.current_stage
        if not self._context:
            self._context = self._fsm.create_context(data)
            if self._transform_context_factory:
                self._context.transform_context_factory = (
                    self._transform_context_factory
                )
        else:
            # Update context data
            if isinstance(self._context.data, dict):
                self._context.data.update(data)
            else:
                self._context.data = data

        result = self._fsm.execute_step_sync(self._context)

        # Sync transform mutations back to caller's data dict.
        # Transforms modify context.data in place; without this sync,
        # the caller won't see new keys set by transforms (e.g.
        # _questions, _artifact_id).
        if isinstance(self._context.data, dict):
            data.update(self._context.data)

        after_stage = self.current_stage

        # Log transition evaluation details
        if before_stage != after_stage:
            # Log the condition that was evaluated
            stage_meta = self._stage_metadata.get(before_stage, {})
            transitions = stage_meta.get("transitions", [])
            for trans in transitions:
                if trans.get("target") == after_stage:
                    condition = trans.get("condition", "unconditional")
                    logger.debug(
                        "WizardFSM transition: '%s' -> '%s' via condition: %s",
                        before_stage,
                        after_stage,
                        condition,
                    )
                    break
        else:
            # Log why no transition occurred
            stage_meta = self._stage_metadata.get(before_stage, {})
            transitions = stage_meta.get("transitions", [])
            if transitions:
                logger.debug(
                    "WizardFSM no transition from '%s': %d transitions defined, none matched",
                    before_stage,
                    len(transitions),
                )
                for trans in transitions:
                    target = trans.get("target", "?")
                    condition = trans.get("condition", "unconditional")
                    logger.debug(
                        "  - target='%s', condition='%s'",
                        target,
                        condition,
                    )

        return result

    async def step_async(self, data: dict[str, Any]) -> StepResult:
        """Execute one FSM step asynchronously with given data.

        Mirrors step() but uses execute_step_async so that async
        pre-tests, transforms, and hooks are properly awaited.

        Args:
            data: Data dict for transition evaluation

        Returns:
            StepResult with transition details
        """
        before_stage = self.current_stage
        if not self._context:
            self._context = self._fsm.create_context(data)
            if self._transform_context_factory:
                self._context.transform_context_factory = (
                    self._transform_context_factory
                )
        else:
            # Update context data
            if isinstance(self._context.data, dict):
                self._context.data.update(data)
            else:
                self._context.data = data

        result = await self._fsm.execute_step_async(self._context)

        # Sync transform mutations back to caller's data dict.
        # Transforms modify context.data in place; without this sync,
        # the caller won't see new keys set by transforms (e.g.
        # _questions, _artifact_id).
        if isinstance(self._context.data, dict):
            data.update(self._context.data)

        after_stage = self.current_stage

        # Log transition evaluation details
        if before_stage != after_stage:
            stage_meta = self._stage_metadata.get(before_stage, {})
            transitions = stage_meta.get("transitions", [])
            for trans in transitions:
                if trans.get("target") == after_stage:
                    condition = trans.get("condition", "unconditional")
                    logger.debug(
                        "WizardFSM transition: '%s' -> '%s' via condition: %s",
                        before_stage,
                        after_stage,
                        condition,
                    )
                    break
        else:
            stage_meta = self._stage_metadata.get(before_stage, {})
            transitions = stage_meta.get("transitions", [])
            if transitions:
                logger.debug(
                    "WizardFSM no transition from '%s': %d transitions defined, none matched",
                    before_stage,
                    len(transitions),
                )
                for trans in transitions:
                    target = trans.get("target", "?")
                    condition = trans.get("condition", "unconditional")
                    logger.debug(
                        "  - target='%s', condition='%s'",
                        target,
                        condition,
                    )

        return result

    def go_back(self, history: list[str]) -> bool:
        """Navigate to previous stage.

        Args:
            history: List of visited stage names

        Returns:
            True if back navigation succeeded
        """
        if len(history) <= 1:
            return False

        if not self.can_go_back():
            return False

        # Get previous stage from history
        previous_stage = history[-2] if len(history) >= 2 else None
        if not previous_stage:
            return False

        # Restore context to previous stage
        if self._context:
            self._context.set_state(previous_stage)
            return True

        return False

    def restart(self) -> None:
        """Reset wizard to start stage.

        Clears the execution context to start fresh.
        """
        self._context = None

    def serialize(self) -> dict[str, Any]:
        """Serialize wizard state for persistence.

        Returns:
            Dict containing serializable wizard state
        """
        return {
            "current_stage": self.current_stage,
            "history": self._get_history(),
            "data": self._get_data(),
        }

    def restore(self, state: dict[str, Any]) -> None:
        """Restore wizard from serialized state.

        Deep-copies `data` to break any shared reference with the
        caller (e.g. ``manager.metadata``).  Without this, transforms
        that mutate ``context.data`` during ``step_async`` would
        contaminate the metadata dict, causing ``json.dumps(metadata)``
        to crash on non-serializable objects.

        Args:
            state: Previously serialized state dict
        """
        current_stage = state.get("current_stage")
        data = copy.deepcopy(state.get("data", {}))

        if current_stage:
            # Create new context with restored state
            self._context = self._fsm.create_context(data)
            self._context.set_state(current_stage)
            if self._transform_context_factory:
                self._context.transform_context_factory = (
                    self._transform_context_factory
                )

    @property
    def start_stage(self) -> str:
        """Name of the wizard's start stage (``is_start`` or first-defined).

        Returns:
            Name of the start stage
        """
        return self._find_start_stage()

    def _find_start_stage(self) -> str:
        """Find the start stage.

        Returns:
            Name of the start stage
        """
        for name, meta in self._stage_metadata.items():
            if meta.get("is_start"):
                return name
        # Fallback to first stage
        return (
            next(iter(self._stage_metadata.keys()))
            if self._stage_metadata
            else "start"
        )

    def _get_history(self) -> list[str]:
        """Get stage history from context.

        Returns:
            List of visited stage names
        """
        if self._context:
            # Try to get from context metadata
            history = self._context.metadata.get("state_history", [])
            if history:
                return list(history)
        return [self.current_stage]

    def _get_data(self) -> dict[str, Any]:
        """Get current data from context.

        Returns:
            Current data dict
        """
        if self._context:
            if isinstance(self._context.data, dict):
                return self._context.data.copy()
        return {}

    # ----------------------------------------------------------------
    # Lifecycle — mirrors AdvancedFSM's six members one for one, because
    # a wrapper that hides what it wraps is how the daemon thread this
    # releases became un-releasable in the first place.
    # ----------------------------------------------------------------

    def close(self) -> None:
        """Release the wrapped FSM's lifecycle resources. Idempotent.

        Stops and joins the daemon event-loop thread that repeated
        synchronous :meth:`step` creates on the underlying
        :class:`AdvancedFSM`, and closes its resource manager. Cascades to
        every registered subflow this FSM **owns** (see
        :meth:`register_subflow`); a subflow registered with ``owns=False``
        belongs to its caller and is left alone.

        Prefer :meth:`aclose` from async code so a resource whose cleanup
        is a coroutine is awaited rather than skipped.

        The FSM remains usable after close: a subsequent synchronous
        :meth:`step` lazily creates a new bridge. That is what makes an
        unconditional teardown — a test fixture, a ``with`` block — safe to
        apply without tracking whether this FSM was ever stepped.
        """
        self._close_subflows()
        self._fsm.close()

    async def aclose(self) -> None:
        """Async counterpart of :meth:`close`.

        Awaits the wrapped FSM's async resource cleanup — so a provider
        exposing an ``aclose`` / ``cleanup`` coroutine is awaited rather
        than skipped — then stops and joins the bridge thread. Cascades to
        owned subflows via *their* :meth:`aclose`.

        Same idempotence and same reusability as :meth:`close`.
        """
        await self._aclose_subflows()
        await self._fsm.aclose()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def _close_subflows(self) -> None:
        """Close owned subflows, isolating each child's failure.

        Teardown is a cascade: one subflow raising must not orphan the
        ones registered after it.
        """
        for name, subflow in list(self._subflow_registry.items()):
            if name not in self._owns_subflows:
                continue
            try:
                subflow.close()
            except Exception:
                logger.exception("Error closing subflow %r", name)

    async def _aclose_subflows(self) -> None:
        """Async counterpart of :meth:`_close_subflows`."""
        for name, subflow in list(self._subflow_registry.items()):
            if name not in self._owns_subflows:
                continue
            try:
                await subflow.aclose()
            except Exception:
                logger.exception("Error closing subflow %r", name)


def create_wizard_fsm(
    fsm_config: dict[str, Any],
    stage_metadata: dict[str, dict[str, Any]],
    custom_functions: dict[str, Callable[..., Any]] | None = None,
    settings: dict[str, Any] | None = None,
    subflow_registry: dict[str, WizardFSM] | None = None,
) -> WizardFSM:
    """Factory function to create a WizardFSM instance.

    Args:
        fsm_config: FSM configuration dict
        stage_metadata: Stage metadata dict
        custom_functions: Optional custom functions to register
        settings: Wizard-level settings dict (optional)
        subflow_registry: Dict mapping subflow names to WizardFSM instances

    Returns:
        Configured WizardFSM instance
    """
    from dataknobs_fsm.api.advanced import create_advanced_fsm

    advanced_fsm = create_advanced_fsm(fsm_config, custom_functions=custom_functions)
    return WizardFSM(
        advanced_fsm, stage_metadata, settings=settings, subflow_registry=subflow_registry
    )
