"""Asynchronous execution engine for FSM processing."""

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any, Dict, List, Tuple

from dataknobs_fsm.core.arc import ArcDefinition, DataIsolationMode, PushArc
from dataknobs_fsm.core.exceptions import FunctionError
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.modes import ProcessingMode
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import StateType
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.common import (
    NetworkSelector,
    TransitionSelectionMode,
    TraversalStrategy
)
from dataknobs_fsm.execution.base_engine import BaseExecutionEngine
from dataknobs_fsm.functions.base import FunctionContext
from dataknobs_fsm.functions.base import ValidationError as FSMValidationError
from dataknobs_fsm.functions.base import as_state_test_callable
from dataknobs_fsm.core.data_wrapper import ensure_dict

logger = logging.getLogger(__name__)


class AsyncExecutionEngine(BaseExecutionEngine):
    """Asynchronous execution engine for FSM.
    
    This engine handles:
    - True async execution of state functions
    - Parallel arc evaluation
    - Async resource management
    - Non-blocking state transitions
    """
    
    def __init__(
        self,
        fsm: FSM,
        strategy: TraversalStrategy = TraversalStrategy.DEPTH_FIRST,
        selection_mode: TransitionSelectionMode = TransitionSelectionMode.HYBRID,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_hooks: bool = True,
        custom_functions: dict[str, Callable] | None = None,
    ):
        """Initialize async execution engine.

        Args:
            fsm: FSM to execute.
            strategy: Traversal strategy for execution.
            selection_mode: Transition selection mode (strategy, scoring, or hybrid).
            max_retries: Maximum retry attempts for recoverable failures.
            retry_delay: Delay between retries in seconds.
            enable_hooks: Enable execution hooks.
            custom_functions: Optional custom functions to merge with FSM registry.
        """
        super().__init__(fsm, strategy, selection_mode, max_retries, retry_delay)
        self.enable_hooks = enable_hooks
        self._pre_transition_hooks: list[Callable] = []
        self._post_transition_hooks: list[Callable] = []
        self._error_hooks: list[Callable] = []
        self._custom_functions: dict[str, Callable] = custom_functions or {}
    
    async def _fire_hooks(self, hooks: list[Callable], *args: Any) -> None:
        """Fire a list of hooks, awaiting async hooks.

        Hooks must not break execution; exceptions are silently ignored.

        Args:
            hooks: List of hook callables.
            *args: Arguments to pass to each hook.
        """
        if not self.enable_hooks:
            return
        for hook in hooks:
            try:
                result = hook(*args)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass  # Hooks must not break execution

    def _get_merged_functions(self) -> dict[str, Any]:
        """Return FSM function registry merged with custom functions.

        Returns:
            Dictionary of all available functions.
        """
        function_registry = getattr(self.fsm, 'function_registry', {})
        if hasattr(function_registry, 'functions'):
            functions = dict(function_registry.functions)
        else:
            functions = dict(function_registry)
        functions.update(self._custom_functions)
        return functions

    def add_pre_transition_hook(self, hook: Callable) -> None:
        """Add a pre-transition hook.

        Args:
            hook: Hook function to add.
        """
        self._pre_transition_hooks.append(hook)

    def add_post_transition_hook(self, hook: Callable) -> None:
        """Add a post-transition hook.

        Args:
            hook: Hook function to add.
        """
        self._post_transition_hooks.append(hook)

    def add_error_hook(self, hook: Callable) -> None:
        """Add an error hook.

        Args:
            hook: Hook function to add.
        """
        self._error_hooks.append(hook)

    async def execute(
        self,
        context: ExecutionContext,
        data: Any = None,
        max_transitions: int = 1000,
        arc_name: str | None = None
    ) -> Tuple[bool, Any]:
        """Execute the FSM asynchronously with given context.
        
        Args:
            context: Execution context.
            data: Input data to process.
            max_transitions: Maximum transitions before stopping.
            arc_name: Optional specific arc name to follow.
            
        Returns:
            Tuple of (success, result).
        """
        start_time = time.time()
        self._execution_count += 1
        
        # Only override context.data if data was explicitly provided
        if data is not None:
            context.data = data
        
        # Enter the initial state on the parent context ONLY for single-record
        # mode. Batch and stream spin up a fresh child context per item/record
        # and route each through ``_enter_and_execute_child`` ->
        # ``_enter_initial_state`` — which runs the child's initial-state entry
        # AND releases its owned resources in ``_execute_single``'s finally.
        # Entering the initial state on the parent *aggregate* context here would
        # allocate the start state's resources on a context that is never run
        # through ``_execute_single``, so they would never be released (a leak on
        # every batch/stream run), and would spuriously run the start-state
        # transform / pre-validators once on the aggregate context. The parent's
        # ``current_state`` is unused by ``_execute_batch`` / ``_execute_stream``
        # (they read ``batch_data`` / ``stream_context`` and operate on
        # children), so skipping its entry is safe.
        if context.data_mode == ProcessingMode.SINGLE:
            success, error = await self._enter_initial_state(context)
            if not success:
                return False, error

        try:
            # Execute based on data mode
            if context.data_mode == ProcessingMode.SINGLE:
                result = await self._execute_single(context, max_transitions, arc_name)
            elif context.data_mode == ProcessingMode.BATCH:
                result = await self._execute_batch(context, max_transitions)
            elif context.data_mode == ProcessingMode.STREAM:
                result = await self._execute_stream(context, max_transitions)
            else:
                result = False, f"Unknown data mode: {context.data_mode}"
                
            self._total_execution_time += time.time() - start_time
            return result
            
        except Exception as e:
            self._error_count += 1
            self._total_execution_time += time.time() - start_time
            return False, str(e)
    
    async def _execute_single(
        self,
        context: ExecutionContext,
        max_transitions: int,
        arc_name: str | None = None
    ) -> Tuple[bool, Any]:
        """Execute in single record mode asynchronously.
        
        Args:
            context: Execution context.
            max_transitions: Maximum transitions.
            arc_name: Optional specific arc name to follow.
            
        Returns:
            Tuple of (success, result).
        """
        transitions = 0

        try:
            while transitions < max_transitions:
                # Check if we're in a final state
                if await self._is_final_state(context.current_state):
                    return self.finalize_single_result(context)

                # Get available transitions
                transitions_available = await self._get_available_transitions(
                    context.current_state,
                    context,
                    arc_name
                )

                if not transitions_available:
                    # No valid transitions - check if this is a final state
                    if await self._is_final_state(context.current_state):
                        return self.finalize_single_result(context)

                    # In a subflow at a final state with no transitions - pop back
                    # to the parent and continue (covers a subflow whose initial
                    # state is itself a final state).
                    if context.network_stack:
                        if await self._check_subflow_completion(context):
                            continue

                    return False, f"No valid transitions from state: {context.current_state}"

                # Choose transition based on strategy
                next_transition = await self._choose_transition(
                    transitions_available,
                    context
                )

                if not next_transition:
                    return False, "No transition selected"

                # Execute transition
                success = await self._execute_transition(
                    next_transition,
                    context
                )

                if not success:
                    return False, f"Transition failed: {next_transition}"

                # If that transition completed a subflow (reached the sub-network's
                # final state), pop back to the parent network now — before the next
                # loop-top final-state check would otherwise finalize the whole run
                # on a sub-network final state (is_final_state_common is global).
                if context.network_stack:
                    await self._check_subflow_completion(context)

                transitions += 1

            return False, f"Maximum transitions ({max_transitions}) exceeded"
        finally:
            # Release the resources owned by the state the run ENDS on. Each
            # state's own ('owned') resources are released as it is left — a
            # regular transition (_execute_transition) or a subflow pop
            # (_pop_subflow) — but the final/terminal state (or a dead-end / max-
            # transitions stop) is never "left", so without this its acquisitions
            # would be stranded until the resource manager is torn down. This
            # completes the "released on every exit path" contract for the run's
            # last state. Idempotent (clears current_state_owned_resources), and
            # touches only this state's own resources — any still-inherited
            # ancestor resources belong to an ancestor that already unwound.
            self._release_owned_state_resources(context)
    
    async def _enter_and_execute_child(
        self,
        child_context: ExecutionContext,
        max_transitions: int,
        arc_name: str | None = None,
    ) -> Tuple[bool, Any]:
        """Enter a fresh child's initial state (parity path) then run it.

        Batch and stream both spin up a fresh child :class:`ExecutionContext`
        per item/record. Routing its initial-state entry through the shared
        :meth:`_enter_initial_state` (rather than a bare ``set_state``) keeps
        each item at parity with single-record execution: its initial-state
        pre-validators run, its initial-state resources are allocated, and its
        initial-state transforms execute. A rejected/failed initial entry
        surfaces as that item's ``(False, error)`` result instead of silently
        skipping the work.
        """
        success, error = await self._enter_initial_state(child_context)
        if not success:
            return False, error
        return await self._execute_single(
            child_context, max_transitions, arc_name
        )

    async def _execute_batch(
        self,
        context: ExecutionContext,
        max_transitions: int
    ) -> Tuple[bool, Any]:
        """Execute in batch mode asynchronously.

        Args:
            context: Execution context.
            max_transitions: Maximum transitions per item.

        Returns:
            Tuple of (success, results).
        """
        if not context.batch_data:
            return False, "No batch data to process"

        # Process items in parallel
        tasks = []
        for i, item in enumerate(context.batch_data):
            # Create child context for this item
            item_context = context.create_child_context(f"batch_{i}")
            item_context.data = item

            # Enter the item's initial state through the shared parity path
            # (pre-validators + resource allocation + initial transforms),
            # then run it — matching single-record execution.
            task = asyncio.create_task(
                self._enter_and_execute_child(item_context, max_transitions)
            )
            tasks.append(task)
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        batch_results = []
        batch_errors = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                batch_errors.append((i, result))
            else:
                # Result is a tuple[bool, Any] at this point
                success, value = result  # type: ignore
                if success:  # success
                    batch_results.append(value)
                else:
                    batch_errors.append((i, Exception(value)))
        
        return len(batch_errors) == 0, {
            'results': batch_results,
            'errors': batch_errors
        }
    
    async def _execute_stream(
        self,
        context: ExecutionContext,
        max_transitions: int
    ) -> Tuple[bool, Any]:
        """Execute in stream mode asynchronously.
        
        Args:
            context: Execution context.
            max_transitions: Maximum transitions per chunk.
            
        Returns:
            Tuple of (success, stream_stats).
        """
        if not context.stream_context:
            return False, "No stream context provided"
        
        chunks_processed = 0
        total_records = 0
        errors = []
        
        # Process each chunk
        while True:
            # Get next chunk from stream
            chunk = context.stream_context.get_next_chunk()
            if not chunk:
                break
            
            context.set_stream_chunk(chunk)
            
            # Process chunk data
            for record in chunk.data:
                record_context = context.create_child_context(
                    f"stream_{chunks_processed}_{total_records}"
                )
                record_context.data = record

                # Enter the record's initial state through the shared parity path
                # (pre-validators + resource allocation + initial transforms),
                # then run it — matching single-record execution.
                success, result = await self._enter_and_execute_child(
                    record_context,
                    max_transitions
                )
                
                if not success:
                    errors.append((total_records, result))
                
                # Merge context
                context.merge_child_context(
                    f"stream_{chunks_processed}_{total_records}"
                )
                
                total_records += 1
            
            chunks_processed += 1
            
            # Check if this was the last chunk
            if chunk.is_last:
                break
        
        return len(errors) == 0, {
            'chunks_processed': chunks_processed,
            'records_processed': total_records,
            'errors': errors
        }
    
    async def _get_available_transitions(
        self,
        state_name: str,
        context: ExecutionContext,
        arc_name: str | None = None
    ) -> List[ArcDefinition]:
        """Get available transitions from current state asynchronously.
        
        This evaluates pre-conditions in parallel.
        
        Args:
            state_name: Current state name.
            context: Execution context.
            arc_name: Optional specific arc name to filter by.
            
        Returns:
            List of available arc definitions.
        """
        network = await self._get_current_network(context)
        if not network or state_name not in network.states:
            return []
        
        state = network.states[state_name]
        available = []
        
        # Filter arcs by name if specified
        arcs_to_evaluate = state.outgoing_arcs
        if arc_name:
            arcs_to_evaluate = [arc for arc in state.outgoing_arcs 
                              if hasattr(arc, 'name') and arc.name == arc_name]
            # If no arcs match the specified name, return empty list
            if not arcs_to_evaluate:
                return []
        
        # Evaluate all arc pre-conditions in parallel
        tasks = [
            (arc, asyncio.create_task(self._evaluate_arc(arc, context)))
            for arc in arcs_to_evaluate
        ]

        # Await ALL evaluations (return_exceptions=True) so a condition that
        # raises never orphans a sibling task ("Task exception was never
        # retrieved"). _evaluate_arc has already converted a soft reject
        # (ValidationError) to False, so any exception surfaced here is a
        # genuine evaluation failure — re-raise it so the record errors out
        # (engine.execute() turns it into a failed record result) instead of
        # being silently de-selected.
        outcomes = await asyncio.gather(
            *(task for _, task in tasks), return_exceptions=True
        )
        for (arc, _), outcome in zip(tasks, outcomes):
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome:
                available.append(arc)

        # Sort by priority (higher first)
        available.sort(key=lambda a: a.priority, reverse=True)

        return available
    
    async def _evaluate_arc(
        self,
        arc: ArcDefinition,
        context: ExecutionContext
    ) -> bool:
        """Evaluate if an arc can be executed.
        
        Args:
            arc: Arc definition.
            context: Execution context.
            
        Returns:
            True if arc can be executed.
        """
        if not arc.pre_test:
            return True

        functions = self._get_merged_functions()

        if arc.pre_test not in functions:
            return False

        # Execute pre-test function. A bare IStateTestFunction instance injected
        # via the engine's custom_functions merge (which, unlike the manager and
        # the config builder, stores functions un-normalized) is not callable —
        # normalize it to its bound .test method so every pre-test is invoked
        # uniformly as func(data, context).
        pre_test_func = as_state_test_callable(functions[arc.pre_test])

        # Acquire the arc's declared resources and inject them (plus the role
        # map) into the condition's function context, so a resource-aware
        # predicate can route on them. The factory stays transform-scoped
        # (apply_factory=False) — its documented contract is for transforms.
        role_bindings = getattr(arc, 'required_resources', None) or {}
        owner_id = self._arc_resource_owner(context, arc)
        arc_label = (
            f"arc '{getattr(context, 'current_state', '?')}->{arc.target_state}' "
            f"condition"
        )
        arc_resources = self._acquire_named_resources(
            context, role_bindings.values(), owner_id, owner_label=arc_label
        )
        try:
            func_context = self._build_function_context(
                context,
                state_name=getattr(context, 'current_state', '') or '',
                function_name=arc.pre_test,
                resources=arc_resources,
                role_bindings=role_bindings,
                base_metadata={
                    'source_state': getattr(context, 'current_state', None),
                    'target_state': arc.target_state,
                    'arc_priority': arc.priority,
                    # Conditions never stream; carried for shape-parity with the
                    # sync condition context (core/arc.py _create_function_context).
                    'stream_enabled': False,
                },
                apply_factory=False,
            )

            try:
                # Check if it's async
                if asyncio.iscoroutinefunction(pre_test_func):
                    result = await pre_test_func(context.data, func_context)
                else:
                    # Run sync function in executor
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None,
                        pre_test_func,
                        context.data,
                        func_context,
                    )
            except FSMValidationError:
                # An explicit "this record is invalid" signal → the arc is
                # unavailable (a soft reject). The build_record_validator gate
                # predicates already return False for invalid data; this branch
                # is for a hand-authored condition that prefers to raise.
                return False
            except Exception:
                # A genuine evaluation failure — a resource the condition needs
                # is missing/down, a validator has a bug, a reference lookup
                # errored. This is NOT a clean "record invalid" outcome, so it
                # must surface as a record error (counted in errors / tripping
                # error_threshold) rather than silently de-selecting the arc and
                # routing the record to the fall-through reject terminal — which
                # would hide an infrastructure outage as a data-quality drop.
                # _get_available_transitions propagates this; engine.execute()
                # converts it to a failed record result.
                logger.warning(
                    "Arc condition %r raised; surfacing as an evaluation error "
                    "(arc %s->%s)",
                    arc.pre_test,
                    getattr(context, 'current_state', '?'),
                    arc.target_state,
                    exc_info=True,
                )
                raise
        finally:
            self._release_named_resources(
                context, arc_resources, owner_id, owner_label=arc_label
            )

        # Handle tuple return from test functions (bool, reason)
        if isinstance(result, tuple):
            return bool(result[0])
        return bool(result)
    
    async def _choose_transition(
        self,
        available: List[ArcDefinition],
        context: ExecutionContext
    ) -> ArcDefinition | None:
        """Choose transition using common transition selector.
        
        Args:
            available: Available transitions.
            context: Execution context.
            
        Returns:
            Selected arc or None.
        """
        return self.transition_selector.select_transition(
            available,
            context,
            strategy=self.strategy
        )
    
    async def _execute_transition(
        self,
        arc: ArcDefinition,
        context: ExecutionContext
    ) -> bool:
        """Execute a state transition asynchronously.

        Mirrors the sync engine's retry/error categorization and hook firing.

        Args:
            arc: Arc to execute.
            context: Execution context.

        Returns:
            True if successful.
        """
        # Check if this is a PushArc (subflow transition). A push arc is not a
        # flat transition: it pushes a sub-network onto the stack, isolates the
        # data, and enters the sub-network's initial state (parity with the
        # sync engine's _execute_transition dispatch).
        if isinstance(arc, PushArc):
            return await self._execute_push_arc(context, arc)

        # Fire pre-transition hooks
        await self._fire_hooks(self._pre_transition_hooks, context, arc)

        # Acquire the arc's declared resources once for this transition; they
        # are injected (by name, plus a role map) into the transform's function
        # context and released in the finally below. Acquired even before the
        # retry loop so the same handles are reused across retries.
        role_bindings = getattr(arc, 'required_resources', None) or {}
        owner_id = self._arc_resource_owner(context, arc)
        arc_label = (
            f"arc '{getattr(context, 'current_state', '?')}->{arc.target_state}'"
        )
        # Acquire only when a transform will actually consume the resources — a
        # resource-bearing arc with no transform has no consumer for the handles.
        arc_resources = (
            self._acquire_named_resources(
                context, role_bindings.values(), owner_id, owner_label=arc_label
            )
            if arc.transform
            else {}
        )
        try:
            retry_count = 0
            while retry_count <= self.max_retries:
                try:
                    # Execute arc transform if defined
                    if arc.transform:
                        functions = self._get_merged_functions()

                        # Normalize to list for uniform handling
                        transform_names = (
                            arc.transform
                            if isinstance(arc.transform, list)
                            else [arc.transform]
                        )

                        # Build the resource-bearing function context once per
                        # attempt — unconditionally, even for a resourceless arc,
                        # so the context type is consistent and the
                        # transform_context_factory is honored on the arc path.
                        func_context = self._build_function_context(
                            context,
                            state_name=getattr(context, 'current_state', '') or '',
                            function_name=(
                                transform_names[0]
                                if transform_names
                                else arc.target_state
                            ),
                            resources=arc_resources,
                            role_bindings=role_bindings,
                            base_metadata={
                                'source_state': getattr(
                                    context, 'current_state', None
                                ),
                                'target_state': arc.target_state,
                                'arc_priority': arc.priority,
                                # Arc transforms run buffered (not streaming);
                                # carried so the transform context shape matches
                                # the condition path and the sync engine
                                # (core/arc.py _create_function_context always
                                # sets stream_enabled in metadata).
                                'stream_enabled': False,
                            },
                        )

                        for transform_name in transform_names:
                            if transform_name not in functions:
                                continue
                            transform_func = functions[transform_name]

                            # Resource-bearing interface transforms (and inline-
                            # code wrappers, which carry interface=ITransformFunction)
                            # must be dispatched deterministically as
                            # (dict, func_context) so injected resources reach them.
                            if self._is_interface_transform(transform_func):
                                result = await self._invoke_state_transform(
                                    transform_func, context, func_context, None
                                )
                            else:
                                # Check if it's async
                                is_async = asyncio.iscoroutinefunction(transform_func)
                                if not is_async and callable(transform_func):
                                    is_async = asyncio.iscoroutinefunction(
                                        transform_func.__call__
                                    )

                                if is_async:
                                    result = await transform_func(
                                        context.data, func_context
                                    )
                                else:
                                    loop = asyncio.get_running_loop()
                                    result = await loop.run_in_executor(
                                        None,
                                        transform_func,
                                        context.data,
                                        func_context,
                                    )

                            context.data = self._coalesce_transform_result(
                                result, context.data
                            )

                    # Leaving the source state: release its owned resources
                    # (held until now so a nested push can inherit them, and so a
                    # transform reading context.current_state_resources can still
                    # see them — the arc transform itself receives the arc's own
                    # acquired resources via func_context, not these), then enter
                    # the target through the shared entry path
                    # so it runs pre-validators and populates
                    # current_state_resources for child inheritance — at parity
                    # with the sync engine's _execute_transition (which calls
                    # enter_state). A rejecting pre-validator fails the
                    # transition (deterministic; not retried).
                    self._release_owned_state_resources(context)
                    if not await self.enter_state(context, arc.target_state):
                        return False

                    self._transition_count += 1

                    # Fire post-transition hooks
                    await self._fire_hooks(
                        self._post_transition_hooks, context, arc
                    )

                    return True

                except (TypeError, AttributeError, ValueError, SyntaxError) as e:
                    # Deterministic errors - don't retry
                    self._error_count += 1
                    await self._fire_hooks(self._error_hooks, context, arc, e)
                    return False

                except FunctionError as e:
                    # Function failure - don't retry
                    self._error_count += 1
                    await self._fire_hooks(self._error_hooks, context, arc, e)
                    return False

                except Exception as e:
                    # Potentially recoverable - retry with backoff
                    self._error_count += 1
                    await self._fire_hooks(self._error_hooks, context, arc, e)
                    retry_count += 1
                    if retry_count <= self.max_retries:
                        await asyncio.sleep(self.retry_delay * retry_count)
                    else:
                        return False

            return False
        finally:
            self._release_named_resources(
                context, arc_resources, owner_id, owner_label=arc_label
            )

    async def _execute_push_arc(
        self,
        context: ExecutionContext,
        push_arc: PushArc,
        max_subflow_depth: int = 10
    ) -> bool:
        """Execute a push arc to transition into a subflow network.

        Drives the shared subflow lifecycle (``base_engine`` helpers): resolves
        and validates the target before committing, isolates the parent data via
        the shared ``DataIsolationMode.apply`` helper (the single source of
        truth for data isolation), records a :class:`SubflowFrame` (for the
        pop's ``result_mapping`` and a failed entry's rollback), and enters the
        sub-network's initial
        state via the async :meth:`enter_state` — which runs pre-validators and
        allocates the sub-network state's own resources, at parity with the sync
        engine. The isolation deepcopy / serialize is offloaded off the event
        loop.

        The sub-network inherits the pushing state's resources on every path:
        the async regular-transition and initial-state entries now also route
        through :meth:`enter_state`, so ``current_state_resources`` (which the
        inheritance seeds from) is populated regardless of how the pushing state
        was entered. Those inherited resources are held through the subflow and
        released for the parent level on pop.

        Args:
            context: Execution context.
            push_arc: PushArc definition with target network and mappings.
            max_subflow_depth: Maximum allowed nesting depth for subflows.

        Returns:
            True if the push was successful.
        """
        if self.subflow_depth_exceeded(context, push_arc, max_subflow_depth):
            return False

        network_name, explicit_initial_state = self.parse_push_target(push_arc)

        target_network = self.fsm.networks.get(network_name)
        if not target_network:
            logger.error("Target network '%s' not found for PushArc", network_name)
            return False

        # Save current state resources so the sub-network inherits them. Every
        # entry path (initial, regular transition, nested push) now routes
        # through enter_state, so this reflects the pushing state's resources
        # regardless of how it was entered.
        parent_state_resources = getattr(context, 'current_state_resources', {})

        # Fire pre-transition hooks
        await self._fire_hooks(self._pre_transition_hooks, context, push_arc)

        # Resolve the target state BEFORE committing the push, so a bad target
        # fails cleanly without having mutated the context.
        target_state = self.resolve_subflow_initial_state(
            target_network, network_name, explicit_initial_state
        )
        if target_state is None:
            return False

        mapped_data = self.prepare_subflow_input(push_arc, context.data)
        isolated_data = await self._isolate_subflow_data(
            push_arc.isolation_mode, mapped_data
        )

        # Commit the push (replace data, push network + frame, set parent
        # resources), then enter the sub-network's initial state.
        self.begin_subflow(
            context, push_arc, network_name, parent_state_resources, isolated_data
        )
        if not await self.enter_state(context, target_state, run_validators=True):
            logger.error("Failed to enter subflow initial state '%s'", target_state)
            self.rollback_push(context)
            return False

        self._transition_count += 1

        # Fire post-transition hooks
        await self._fire_hooks(self._post_transition_hooks, context, push_arc)

        logger.debug(
            "Pushed to subflow network '%s', state '%s'",
            network_name,
            target_state
        )

        return True

    @staticmethod
    async def _isolate_subflow_data(
        isolation_mode: DataIsolationMode, data: Any
    ) -> Any:
        """Produce the sub-network's isolated data view without stalling the loop.

        ``REFERENCE`` is a pass-through (no work, no thread hop). ``COPY`` /
        ``SERIALIZE`` do a synchronous ``deepcopy`` / JSON round-trip whose cost
        grows with the payload, so they are offloaded to a worker thread — a
        large subflow payload must not block the shared event loop. The actual
        transformation stays the shared ``DataIsolationMode.apply`` helper, so
        no isolation logic forks onto the async path.
        """
        if isolation_mode is DataIsolationMode.REFERENCE:
            return isolation_mode.apply(data)
        return await asyncio.to_thread(isolation_mode.apply, data)

    async def _check_subflow_completion(
        self,
        context: ExecutionContext
    ) -> bool:
        """Pop back to the parent for every subflow now at a final state.

        Async counterpart to ``ExecutionEngine._check_subflow_completion``.
        Drains nested completions in a loop: if popping one level lands on a
        state that is itself a final state of the next network on the stack
        (e.g. a push arc whose ``return_state`` is a final state of the parent
        sub-network), that level pops too, until the current state is no longer
        a subflow final state. Because :meth:`is_final_state_common` treats a
        sub-network final state as "final" for the whole FSM, the main loop must
        call this immediately after each transition (before the next loop-top
        final-state check), and a single-level pop would leave a nested chain
        half-unwound and finalize the run prematurely.

        Args:
            context: Execution context.

        Returns:
            True if at least one subflow level was completed and popped.
        """
        popped_any = False
        while self.subflow_at_final_state(context):
            if not await self._pop_subflow(context):
                break
            popped_any = True
        return popped_any

    async def _pop_subflow(
        self,
        context: ExecutionContext
    ) -> bool:
        """Pop from a subflow back to the parent network.

        Async counterpart to ``ExecutionEngine._pop_subflow``. Consumes the
        :class:`SubflowFrame` recorded at push time to restore the parent's
        inherited-resource view and apply the originating push arc's
        ``result_mapping``, then enters the parent's return state via the async
        :meth:`enter_state`.

        Args:
            context: Execution context.

        Returns:
            True if the pop was successful.
        """
        if not context.network_stack:
            return False

        # Release the sub-network's current-state owned resources before
        # unwinding — done while current_state still names the subflow's final
        # state so the owner key resolves correctly. Inherited (parent-owned)
        # resources are untouched.
        self._release_owned_state_resources(context)

        frame = context.pop_subflow_frame()
        _network_name, return_state = context.pop_network()

        # Release the pushing state's own resources now that the parent has left
        # it for return_state (they were held through the subflow because the
        # sub-network inherited them).
        if frame is not None and frame.pushing_state_owner:
            self._release_owner(context, frame.pushing_state_owner)

        # Restore the parent level's inherited-resource view, then map the
        # sub-network's result back onto the parent data (before the return
        # state's transforms run, so they see the mapped data).
        self.restore_after_pop(context, frame)
        self.apply_subflow_result_mapping(context, frame)

        # Transition to return state if specified
        if return_state:
            if not await self.enter_state(context, return_state, run_validators=True):
                logger.error("Failed to enter return state '%s'", return_state)
                return False

        logger.debug(
            "Popped from subflow, returned to state '%s'",
            return_state or context.current_state
        )

        return True

    async def enter_state(
        self,
        context: ExecutionContext,
        state_name: str,
        run_validators: bool = True,
    ) -> bool:
        """Enter a state on the async engine, at parity with the sync engine.

        Mirrors ``ExecutionEngine.enter_state``: set the state, allocate its
        resources (merging parent-inherited resources) and store them on the
        context for child inheritance, optionally run pre-validators (a failing
        validator releases the just-acquired resources and returns ``False``),
        then run the state's transforms with those resources.

        This closes the async-vs-sync state-entry gap on the subflow push and
        pop-return paths, where the sync engine ran pre-validators and allocated
        state resources but the async engine previously only did ``set_state`` +
        transforms (so sub-network pre-validators never ran and
        ``current_state_resources`` was never populated for child inheritance).

        Args:
            context: Execution context.
            state_name: Name of the state to enter.
            run_validators: Whether to run pre-validators (default True).

        Returns:
            True if state entry succeeded, False if a pre-validator rejected it.
        """
        context.set_state(state_name)
        return await self._establish_state(context, state_name, run_validators)

    async def _establish_state(
        self,
        context: ExecutionContext,
        state_name: str,
        run_validators: bool = True,
    ) -> bool:
        """Run a state's setup, assuming ``context.current_state`` is already it.

        The body of :meth:`enter_state` minus the ``set_state`` call: allocate
        the state's resources (with parent inheritance), optionally run
        pre-validators, then run its transforms. Used directly by
        :meth:`_enter_initial_state` for an already-current (pre-set) initial
        state, so re-establishing it does not record a duplicate history entry.
        """
        state_def = self.fsm.get_state(state_name)
        state_resources, owned = self._allocate_state_resources_with_inheritance(
            context, state_def
        )
        context.current_state_resources = state_resources
        # Hold the just-acquired ('owned') subset so a nested push can inherit it;
        # it is released by _release_owned_state_resources when this state is left
        # (a regular transition to the next state, or a subflow pop).
        context.current_state_owned_resources = owned

        if run_validators and state_def is not None:
            if not await self._run_pre_validators(
                context, state_def, state_name, state_resources
            ):
                # Release only what this entry acquired (not the inherited
                # resources, which the parent owns), then fail the entry.
                self._release_named_resources(
                    context,
                    owned,
                    self._state_resource_owner(context, state_def),
                    owner_label=f"state '{getattr(state_def, 'name', '?')}'",
                )
                context.current_state_resources = {}
                context.current_state_owned_resources = {}
                # Match the sync engine: only set last_error if a more specific
                # upstream error has not already been recorded (don't clobber it).
                if not hasattr(context, 'last_error') or not context.last_error:
                    context.last_error = (
                        f"Pre-validation failed for state '{state_name}'"
                    )
                return False

        # Reuse the resources allocated here (don't re-acquire / release).
        await self._execute_state_transforms(
            context, state_resources=state_resources
        )
        return True

    def _allocate_state_resources_with_inheritance(
        self,
        context: ExecutionContext,
        state_def: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Allocate a state's resources, inheriting the parent state's.

        Mirrors the sync engine's ``_allocate_state_resources``: start from the
        parent-inherited resources, then acquire only the state's own
        requirements that are not already inherited.

        Returns:
            ``(merged_resources, owned_resources)`` — the full view stored on
            the context (parent-inherited + own), and just the resources this
            call acquired (so a failed entry can release exactly those without
            touching the parent-owned ones).
        """
        parent = getattr(context, 'parent_state_resources', None)
        resources: Dict[str, Any] = dict(parent) if parent else {}
        requirements = (
            getattr(state_def, 'resource_requirements', None)
            if state_def is not None
            else None
        )
        if not requirements:
            return resources, {}

        names: list[str] = []
        timeouts: Dict[str, Any] = {}
        for resource_config in requirements:
            name = getattr(resource_config, 'name', None)
            if not name or name in resources:
                # Skip resources already inherited from the parent state.
                continue
            names.append(name)
            # Default to 30s when no timeout is configured, matching the sync
            # engine (ExecutionEngine._allocate_state_resources) rather than
            # leaving it None (an unbounded acquire wait).
            timeouts[name] = (
                getattr(resource_config, 'timeout_seconds', None)
                or getattr(resource_config, 'timeout', None)
                or 30
            )

        owned = self._acquire_named_resources(
            context,
            names,
            self._state_resource_owner(context, state_def),
            timeouts=timeouts,
            owner_label=f"state '{getattr(state_def, 'name', '?')}'",
        )
        resources.update(owned)
        return resources, owned

    async def _run_pre_validators(
        self,
        context: ExecutionContext,
        state_def: Any,
        state_name: str,
        state_resources: Dict[str, Any],
    ) -> bool:
        """Run a state's pre-validators (awaiting async ones).

        Mirrors ``ExecutionEngine._execute_pre_validators``: a validator
        returning ``False`` fails entry; a dict result is merged into
        ``context.data``; any exception fails validation. Closes the gap where
        the async engine never ran ``pre_validation_functions``.

        Args:
            context: Execution context.
            state_def: The state definition whose pre-validators to run.
            state_name: Name of the state (for the function context).
            state_resources: Resources to inject into the validator context.

        Returns:
            True if all pre-validators pass, False otherwise.
        """
        validators = getattr(state_def, 'pre_validation_functions', None)
        if not validators:
            return True

        for validator_func in validators:
            try:
                func_context = self._build_function_context(
                    context,
                    state_name=state_name,
                    function_name=getattr(validator_func, '__name__', 'validate'),
                    resources=state_resources,
                    base_metadata={'state': state_name, 'phase': 'pre_validation'},
                    # Validators are not transforms — the transform context
                    # factory's documented scope (parity with the sync engine,
                    # whose pre-validators build a plain FunctionContext).
                    apply_factory=False,
                )
                result = validator_func(ensure_dict(context.data), func_context)
                if inspect.isawaitable(result):
                    result = await result
                if result is False:
                    return False
                if isinstance(result, dict):
                    context.data.update(result)
            except Exception:
                # Any error in a pre-validator fails validation (parity w/ sync).
                return False
        return True

    async def _execute_state_transforms(
        self,
        context: ExecutionContext,
        state_resources: Dict[str, Any] | None = None,
    ) -> None:
        """Execute state functions (validators and transforms) when in a state.

        This should be called before evaluating arc conditions to ensure
        that state functions can update the data that conditions depend on.

        Args:
            context: Execution context.
            state_resources: Pre-acquired, caller-owned state resources. When
                provided (by :meth:`enter_state`, which allocates them with
                parent inheritance and stores them on the context for child
                inheritance — parity with the sync engine), this method reuses
                them and does NOT release them; the caller owns their lifecycle.
                When ``None`` (the regular-transition / initial-state callers),
                this method acquires and releases the state's own resources
                itself, the historical behavior.
        """
        network = await self._get_current_network(context)
        if not network or context.current_state not in network.states:
            return

        state = network.states[context.current_state]
        state_name = context.current_state

        # Use base class logic to prepare transforms
        transform_functions, state_obj = self.prepare_state_transform(state, context)

        # Execute validation functions first (async-specific)
        if hasattr(state, 'validation_functions') and state.validation_functions:
            for validator in state.validation_functions:
                try:
                    # Handle both async and sync validators
                    if asyncio.iscoroutinefunction(validator.validate):
                        # Try with state object first (for inline lambdas)
                        try:
                            result = await validator.validate(state_obj)
                        except (TypeError, AttributeError):
                            # Fall back to standard signature
                            result = await validator.validate(ensure_dict(context.data), context)
                    else:
                        # Run sync function in executor
                        loop = asyncio.get_running_loop()
                        try:
                            result = await loop.run_in_executor(None, validator.validate, state_obj)
                        except (TypeError, AttributeError):
                            # Fall back to standard signature
                            result = await loop.run_in_executor(None, validator.validate, ensure_dict(context.data), context)

                    if isinstance(result, dict):
                        # Merge validation results into context data
                        context.data.update(result)
                except Exception:
                    # Log but don't fail - validators are optional
                    pass

        if not transform_functions:
            return

        logger.debug(
            f"Executing {len(transform_functions)} transform functions for state {state_name}"
        )

        # Acquire the state's declared resources and inject them into the
        # function context so resource-bearing transforms (e.g. DatabaseUpsert)
        # can reach them via context.resources. When the caller pre-acquired
        # them (enter_state), reuse and don't release here — the caller owns
        # their lifecycle (mirrors the arc-resource ownership pattern).
        owns_resources = state_resources is None
        if owns_resources:
            state_resources = self._acquire_state_resources(context, state)
        try:
            for transform_func in transform_functions:
                # Skip running transforms on a record that already failed an
                # upstream transform — don't mutate/persist indeterminate data
                # (e.g. don't run the ETL load upsert after the transform
                # raised). Traversal still continues so the record reaches a
                # final state and is reported as a failure. A state declared
                # run_on_failure=True (recovery/cleanup/dead-letter) is exempt
                # and runs regardless.
                if self.should_skip_state_transforms(context, state):
                    logger.debug(
                        "Skipping transform in state '%s': record already "
                        "failed in %s",
                        state_name,
                        self.failed_states_sorted(context),
                    )
                    break
                try:
                    func_context = self._build_function_context(
                        context,
                        state_name=state_name,
                        function_name=getattr(transform_func, '__name__', 'transform'),
                        resources=state_resources,
                        base_metadata={'state': state_name},
                    )

                    result = await self._invoke_state_transform(
                        transform_func, context, func_context, state_obj
                    )

                    # Process result using base class logic
                    self.process_transform_result(result, context, state_name)

                except Exception as e:
                    # Handle error using base class logic
                    self.handle_transform_error(e, context, state_name)
        finally:
            # Release only resources we acquired; caller-owned (pre-acquired)
            # resources are released by the caller (parity with the sync engine,
            # which keeps state resources allocated for child inheritance).
            if owns_resources:
                self._release_state_resources(context, state, state_resources)

    @staticmethod
    def _coalesce_transform_result(result: Any, current_data: Any) -> Any:
        """Resolve an arc transform's return into the next ``context.data``.

        Matches ``ArcExecution._execute_single_transform_async``'s contract:
        unwrap an ``ExecutionResult`` (success → its data; failure → raise
        ``FunctionError``), and treat a ``None`` return as an in-place mutation
        (preserve the input data). Applied uniformly to interface and
        non-interface arc transforms.
        """
        from dataknobs_fsm.functions.base import ExecutionResult

        if isinstance(result, ExecutionResult):
            if not result.success:
                raise FunctionError(result.error or "Transform failed")
            result = result.data
        if result is None:
            return current_data
        return result

    @staticmethod
    def _is_interface_transform(transform_func: Any) -> bool:
        """Whether a transform is an ITransformFunction (raw or wrapped).

        Interface transforms take ``(data, context)`` and read resources off
        ``context.resources``. They must be dispatched deterministically with
        that signature — a ``state_obj``-first call would succeed (one
        positional arg) and silently bypass resource injection.
        """
        from dataknobs_fsm.functions.base import ITransformFunction

        return (
            isinstance(transform_func, ITransformFunction)
            or getattr(transform_func, 'interface', None) is ITransformFunction
        )

    async def _invoke_state_transform(
        self,
        transform_func: Any,
        context: ExecutionContext,
        func_context: FunctionContext,
        state_obj: Any,
    ) -> Any:
        """Invoke a single state transform with the resource-bearing context.

        Interface transforms (``ITransformFunction`` / ``InterfaceWrapper``)
        are dispatched deterministically as ``(dict, func_context)`` so the
        injected resources reach them. Non-interface callables (inline lambdas
        expecting a state object) keep the historical ``state_obj``-first
        dispatch with a ``(data, context)`` fallback for regression safety.
        """
        actual_func = transform_func
        if hasattr(transform_func, 'transform'):
            actual_func = transform_func.transform

        # Detect async via the callable, its __call__, or a wrapper hint.
        is_async = asyncio.iscoroutinefunction(actual_func)
        if not is_async and callable(actual_func):
            # A callable *object* may carry an async ``__call__``; inspect the
            # type's bound slot (``callable()`` above guarantees it exists).
            is_async = asyncio.iscoroutinefunction(type(actual_func).__call__)
        if not is_async and getattr(transform_func, '_is_async', False):
            is_async = True

        if self._is_interface_transform(transform_func):
            data = ensure_dict(context.data)
            if is_async:
                return await actual_func(data, func_context)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, actual_func, data, func_context)

        # Non-interface callable: state_obj first, fall back to (data, context).
        if is_async:
            try:
                return await actual_func(state_obj)
            except (TypeError, AttributeError):
                return await actual_func(ensure_dict(context.data), func_context)
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, actual_func, state_obj)
        except (TypeError, AttributeError):
            return await loop.run_in_executor(
                None, actual_func, ensure_dict(context.data), func_context
            )

    def _build_function_context(
        self,
        context: ExecutionContext,
        *,
        state_name: str,
        function_name: Any,
        resources: Dict[str, Any] | None = None,
        role_bindings: Dict[str, str] | None = None,
        base_metadata: Dict[str, Any] | None = None,
        apply_factory: bool = True,
    ) -> Any:
        """Build the per-invocation function context for state and arc paths.

        Single builder shared by the state-transform, arc-transform, and
        arc-condition paths so they carry an identical context shape:
        name-keyed ``resources``, a ``resource_roles`` role map, the shared
        ``variables``, and the current ``network_name``.

        When ``apply_factory`` is True (transform paths) and the
        ``ExecutionContext`` carries a ``transform_context_factory``, the built
        ``FunctionContext`` is passed through the factory so a consumer's
        roll-your-own context wraps it — honoring that documented hook on the
        async engine for the first time. Arc *condition* (pre-test) contexts
        pass ``apply_factory=False``: they receive resources + roles but the
        factory stays transform-scoped (its documented contract).

        Args:
            context: The owning execution context.
            state_name: State the function runs in (source state for arcs).
            function_name: Representative function name for the context.
            resources: Name-keyed resources to inject.
            role_bindings: ``{role: name}`` map exposed as
                ``metadata['resource_roles']``.
            base_metadata: Path-specific metadata merged into the context.
            apply_factory: Whether to run ``transform_context_factory``.

        Returns:
            A ``FunctionContext`` (default) or the factory's output.
        """
        metadata = dict(base_metadata or {})
        metadata['resource_roles'] = dict(role_bindings or {})
        network_stack = getattr(context, 'network_stack', None)
        func_context = FunctionContext(
            state_name=state_name,
            function_name=function_name,
            metadata=metadata,
            resources=resources or {},
            variables=getattr(context, 'variables', {}) or {},
            network_name=network_stack[-1][0] if network_stack else None,
        )
        factory = getattr(context, 'transform_context_factory', None)
        if apply_factory and factory:
            return factory(func_context)
        return func_context

    def _acquire_named_resources(
        self,
        context: ExecutionContext,
        names: Any,
        owner_id: str,
        *,
        timeouts: Dict[str, Any] | None = None,
        owner_label: str = "owner",
    ) -> Dict[str, Any]:
        """Acquire resources by name into a ``{name: resource}`` mapping.

        Path-agnostic core shared by the state path (state
        ``resource_requirements``) and the arc path (arc
        ``required_resources.values()``). The acquire is a cheap in-process
        bookkeeping call (the async database resource opens its transport
        lazily on first ``await``), so it does not block the loop. Failures are
        logged (not swallowed silently) and skipped.

        Args:
            context: The owning execution context.
            names: Iterable of resource names to acquire.
            owner_id: Ownership key for the acquisitions.
            timeouts: Optional per-name acquire timeouts.
            owner_label: Human-readable owner descriptor for log context.

        Returns:
            ``{name: resource}`` for every name acquired.
        """
        resources: Dict[str, Any] = {}
        resource_manager = getattr(context, 'resource_manager', None)
        if not resource_manager:
            return resources
        timeouts = timeouts or {}
        for name in names:
            if not name or name in resources:
                continue
            try:
                resources[name] = resource_manager.acquire(
                    name=name, owner_id=owner_id, timeout=timeouts.get(name)
                )
            except Exception as e:
                logger.warning(
                    "Failed to acquire resource '%s' for %s: %s",
                    name,
                    owner_label,
                    e,
                )
        return resources

    def _release_named_resources(
        self,
        context: ExecutionContext,
        resources: Dict[str, Any],
        owner_id: str,
        *,
        owner_label: str = "owner",
    ) -> None:
        """Release resources acquired by :meth:`_acquire_named_resources`."""
        if not resources:
            return
        resource_manager = getattr(context, 'resource_manager', None)
        if not resource_manager:
            return
        for name in resources:
            try:
                resource_manager.release(name, owner_id)
            except Exception as e:
                logger.warning(
                    "Failed to release resource '%s' for %s: %s",
                    name,
                    owner_label,
                    e,
                )

    def _release_owner(self, context: ExecutionContext, owner_id: str) -> None:
        """Release every resource acquired under ``owner_id``.

        Used on subflow pop to release the pushing state's acquisitions, whose
        individual names the pop site no longer has — only the stable owner key
        recorded on the frame at push time.
        """
        resource_manager = getattr(context, 'resource_manager', None)
        if not resource_manager:
            return
        try:
            resource_manager.release_all(owner_id)
        except Exception as e:
            logger.warning(
                "Failed to release resources for owner '%s': %s", owner_id, e
            )

    def _acquire_state_resources(
        self,
        context: ExecutionContext,
        state: Any,
    ) -> Dict[str, Any]:
        """Acquire the resources a state declares, keyed by resource name.

        Mirrors the sync engine's resource allocation so async state
        transforms get the same resource-injection contract. Delegates the
        acquire loop to :meth:`_acquire_named_resources`.
        """
        requirements = getattr(state, 'resource_requirements', None)
        if not requirements:
            return {}
        names: list[str] = []
        timeouts: Dict[str, Any] = {}
        for resource_config in requirements:
            name = getattr(resource_config, 'name', None)
            if not name:
                continue
            names.append(name)
            # Default to 30s when no timeout is configured, matching the sync
            # engine rather than leaving it None (an unbounded acquire wait).
            timeouts[name] = (
                getattr(resource_config, 'timeout_seconds', None)
                or getattr(resource_config, 'timeout', None)
                or 30
            )
        return self._acquire_named_resources(
            context,
            names,
            self._state_resource_owner(context, state),
            timeouts=timeouts,
            owner_label=f"state '{getattr(state, 'name', '?')}'",
        )

    def _release_state_resources(
        self,
        context: ExecutionContext,
        state: Any,
        resources: Dict[str, Any],
    ) -> None:
        """Release resources acquired by :meth:`_acquire_state_resources`."""
        self._release_named_resources(
            context,
            resources,
            self._state_resource_owner(context, state),
            owner_label=f"state '{getattr(state, 'name', '?')}'",
        )

    @staticmethod
    def _state_resource_owner(context: ExecutionContext, state: Any) -> str:
        """Build the resource-ownership key for a state's acquisitions."""
        return BaseExecutionEngine._state_resource_owner_for_name(
            context, getattr(state, 'name', 'unknown')
        )

    def _release_owned_state_resources(self, context: ExecutionContext) -> None:
        """Release the current state's owned resources when leaving it.

        A state holds the resources it acquired itself ('owned', excluding any
        parent-inherited ones) while it is active, so a nested push can inherit
        them. On exit — a regular transition to the next state, or a subflow pop
        — those owned resources must be released; the parent-inherited ones are
        left untouched (the ancestor that acquired them owns their lifecycle).

        This wires the release-on-exit half of the state-resource lifecycle that
        ``ExecutionEngine.exit_state`` intended but was never called for, so the
        unified async engine neither leaks across transitions nor strands a
        subflow's resources on pop.
        """
        owned = getattr(context, 'current_state_owned_resources', None)
        if owned:
            self._release_named_resources(
                context,
                owned,
                self._state_resource_owner_for_name(
                    context, context.current_state
                ),
                owner_label=f"state '{context.current_state}'",
            )
        context.current_state_owned_resources = {}

    @staticmethod
    def _arc_resource_owner(context: ExecutionContext, arc: ArcDefinition) -> str:
        """Build the resource-ownership key for an arc's acquisitions.

        Mirrors the sync ``ArcExecution`` owner-id shape so the two engines key
        arc acquisitions the same way.
        """
        source = getattr(arc, 'source_state', None) or getattr(
            context, 'current_state', 'unknown'
        )
        arc_identifier = f"{source}_to_{arc.target_state}"
        execution_id = getattr(context, 'execution_id', 'unknown')
        return f"arc_{arc_identifier}_{execution_id}"

    async def _enter_initial_state(
        self, context: ExecutionContext
    ) -> tuple[bool, str | None]:
        """Ensure the initial state is entered and its transforms executed.

        Handles both cases:
        - current_state not set: find initial state, set it, run transforms
        - current_state pre-set (by ContextFactory): run transforms for it

        Returns:
            (True, None) on success, (False, error_message) on failure.
        """
        # Enter the start state through the shared entry path (pre-validators +
        # current_state_resources population for child inheritance), at parity
        # with the sync engine's _enter_initial_state. When the state is already
        # current (pre-set by a ContextFactory), establish it without a second
        # set_state so it is not recorded twice in the history.
        if not context.current_state:
            initial_state = await self._find_initial_state()
            if not initial_state:
                return False, "No initial state found"
            if not await self.enter_state(context, initial_state):
                return False, self._initial_entry_error(context, initial_state)
        else:
            if not await self._establish_state(context, context.current_state):
                return False, self._initial_entry_error(
                    context, context.current_state
                )
        return True, None

    @staticmethod
    def _initial_entry_error(context: ExecutionContext, state_name: str) -> str:
        """Build the initial-state entry failure message.

        Prefer the specific upstream reason recorded on ``context.last_error``
        (e.g. "Pre-validation failed for state 'X'", set by ``_establish_state``)
        over the generic "Failed to enter initial state 'X'" so a rejecting
        initial-state pre-validator surfaces *why* it rejected, not just that
        entry failed.
        """
        specific = getattr(context, 'last_error', None)
        return specific or f"Failed to enter initial state '{state_name}'"

    async def _find_initial_state(self) -> str | None:
        """Find initial state in FSM.

        Returns:
            Initial state name or None.
        """
        # Use base class implementation (it's synchronous but that's fine)
        return self.find_initial_state_common()
    
    async def _is_final_state(self, state_name: str | None) -> bool:
        """Check if state is a final state.

        Args:
            state_name: Name of state to check.

        Returns:
            True if final state.
        """
        # Use base class implementation
        return self.is_final_state_common(state_name)

    async def _is_final_state_legacy(self, state_name: str | None) -> bool:
        """Legacy implementation kept for reference."""
        if not state_name:
            return False

        # Get the main network - could be a string or object
        main_network_ref = getattr(self.fsm, 'main_network', None)

        if main_network_ref is None:
            # If no main network specified, check all networks
            for network in self.fsm.networks.values():
                if state_name in network.states:
                    state = network.states[state_name]
                    if state.is_end_state() if hasattr(state, 'is_end_state') else state.type == StateType.END:
                        return True
            return False

        # Handle case where main_network is already a network object (FSM wrapper)
        if hasattr(main_network_ref, 'states'):
            main_network = main_network_ref
        # Handle case where main_network is a string (core FSM)
        elif isinstance(main_network_ref, str) and main_network_ref in self.fsm.networks:
            main_network = self.fsm.networks[main_network_ref]
        else:
            return False
        
        # Check if the state exists and is an end state
        if state_name in main_network.states:
            state = main_network.states[state_name]
            return state.is_end_state() if hasattr(state, 'is_end_state') else state.type == StateType.END
        
        return False
    
    async def _get_current_network(
        self,
        context: ExecutionContext
    ) -> StateNetwork | None:
        """Get the current network from context using common network selector.
        
        Args:
            context: Execution context.
            
        Returns:
            Current network or None.
        """
        # Use intelligent selection for async engine by default
        return NetworkSelector.get_current_network(
            self.fsm,
            context,
            enable_intelligent_selection=True
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary of statistics.
        """
        return {
            'execution_count': self._execution_count,
            'transition_count': self._transition_count,
            'error_count': self._error_count,
            'total_execution_time': self._total_execution_time,
            'average_execution_time': (
                self._total_execution_time / self._execution_count
                if self._execution_count > 0 else 0.0
            ),
            'hooks_enabled': self.enable_hooks,
        }
