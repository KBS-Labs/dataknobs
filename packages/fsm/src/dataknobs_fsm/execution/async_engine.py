"""Asynchronous execution engine for FSM processing."""

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any, Dict, List, Tuple

from dataknobs_fsm.core.arc import ArcDefinition
from dataknobs_fsm.core.exceptions import FunctionError
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.core.modes import ProcessingMode
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import StateType
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.engine import TraversalStrategy
from dataknobs_fsm.execution.common import (
    NetworkSelector,
    TransitionSelectionMode
)
from dataknobs_fsm.execution.base_engine import BaseExecutionEngine
from dataknobs_fsm.functions.base import FunctionContext
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
        
        # Enter initial state (handles both pre-set and not-set cases)
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

            transitions += 1
        
        return False, f"Maximum transitions ({max_transitions}) exceeded"
    
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
            
            # Reset to initial state for each item
            initial_state = await self._find_initial_state()
            if initial_state:
                item_context.set_state(initial_state)
            
            # Create task for this item
            task = asyncio.create_task(
                self._execute_single(item_context, max_transitions)
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
                
                # Reset to initial state
                initial_state = await self._find_initial_state()
                if initial_state:
                    record_context.set_state(initial_state)
                
                # Execute for this record
                success, result = await self._execute_single(
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
        tasks = []
        for arc in arcs_to_evaluate:
            task = asyncio.create_task(self._evaluate_arc(arc, context))
            tasks.append((arc, task))
        
        # Wait for all evaluations
        for arc, task in tasks:
            can_execute = await task
            if can_execute:
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

        # Execute pre-test function
        pre_test_func = functions[arc.pre_test]

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
                },
                apply_factory=False,
            )

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
        arc_resources = self._acquire_named_resources(
            context, role_bindings.values(), owner_id, owner_label=arc_label
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

                    # Update state (history is automatically tracked by set_state)
                    context.set_state(arc.target_state)

                    # Execute state transforms when entering the new state
                    await self._execute_state_transforms(context)

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
    
    async def _execute_state_transforms(
        self,
        context: ExecutionContext
    ) -> None:
        """Execute state functions (validators and transforms) when in a state.

        This should be called before evaluating arc conditions to ensure
        that state functions can update the data that conditions depend on.

        Args:
            context: Execution context.
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
        # can reach them via context.resources. Released in the finally block.
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
            self._release_state_resources(context, state, state_resources)

    @staticmethod
    def _coalesce_transform_result(result: Any, current_data: Any) -> Any:
        """Resolve an arc transform's return into the next ``context.data``.

        Mirrors the sync ``ArcExecution._execute_single_transform`` contract:
        unwrap an ``ExecutionResult`` (success → its data; failure → raise
        ``FunctionError``), and treat a ``None`` return as an in-place mutation
        (preserve the input data). Applied uniformly to interface and
        non-interface arc transforms so both engines behave identically.
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
            timeouts[name] = getattr(resource_config, 'timeout_seconds', None) or getattr(
                resource_config, 'timeout', None
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
        state_name = getattr(state, 'name', 'unknown')
        execution_id = getattr(context, 'execution_id', 'unknown')
        return f"state_{state_name}_{execution_id}"

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
        if not context.current_state:
            initial_state = await self._find_initial_state()
            if not initial_state:
                return False, "No initial state found"
            context.set_state(initial_state)

        # Always run transforms for the start state
        await self._execute_state_transforms(context)
        return True, None

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
