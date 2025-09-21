# Implementation Plan: Pre-Transform Validators, State Resources, and Shared Variables

## Overview

This document outlines the implementation plan for four related enhancements to the FSM package:

1. **Pre-Transform Validators**: Validators that run immediately upon entering a state to verify incoming data meets requirements before transforms are applied.

2. **State Resource Allocation**: Proper allocation of resources defined at the state level, making them available to all state functions (pre-validators, transforms, and post-validators).

3. **Arc Resource Merging**: Ensuring arc resources are additive to state resources, preventing conflicts and unnecessary duplication.

4. **Shared Variables**: Enable cross-state communication through shared variables accessible via FunctionContext, allowing states to cache data and share information beyond the primary data payload.

## Current State

### Existing Validator Behavior
- **Location**: Validators currently execute in `_execute_state_functions()`
- **Timing**: After transforms, before arc evaluation
- **Purpose**: Validate transformed data before evaluating transition conditions
- **Storage**: Stored in `state.validation_functions` list

### Execution Flow (Current)
1. Enter state → Execute transforms (`_execute_state_transforms`)
2. Before arc evaluation → Execute validators (`_execute_state_functions`)
3. Evaluate arc conditions
4. Execute arc and transition

## Proposed Changes

### New Execution Flow
1. Enter state → Execute **pre-validators** (NEW)
2. If pre-validation passes → Execute transforms
3. Before arc evaluation → Execute **post-validators** (existing validators)
4. Evaluate arc conditions
5. Execute arc and transition

### Terminology
- **Pre-validators**: New validators that run before transforms
- **Post-validators** (or just "validators"): Existing validators that run after transforms

## Additional Discovery: State Resource Allocation

During documentation, we discovered that while states can define `resources` in their configuration, these resources are NOT currently allocated for state functions. The execution engine passes an empty resources dict `{}` to state transforms.

### Current State Resource Situation
- States CAN define `resources: ["db", "cache"]` in configuration
- StateDefinition HAS `resource_requirements` field
- But `_execute_state_transforms()` creates FunctionContext with `resources={}`
- Only arc transforms currently get allocated resources

This needs to be fixed alongside the pre-validator implementation.

NOTE: A detail that must be considered and resolved is how resources are managed through the execution of subnetworks through push arcs.

## Implementation Checklist

### 1. Schema Updates
- [ ] **File**: `src/dataknobs_fsm/config/schema.py`
  - [ ] Add `pre_validators: List[FunctionReference] = Field(default_factory=list)` to `StateConfig` class (line ~134)
  - [ ] Keep existing `validators` field for backward compatibility (becomes post-validators)

### 2. State Definition Updates
- [ ] **File**: `src/dataknobs_fsm/core/state.py`
  - [ ] Add `pre_validation_functions: List[Callable] = field(default_factory=list)` to `StateDefinition` class
  - [ ] Rename `validation_functions` to `post_validation_functions` (with deprecation warning)
  - [ ] Add property `validation_functions` that returns `post_validation_functions` for backward compatibility

### 3. Config Builder Updates
- [ ] **File**: `src/dataknobs_fsm/config/builder.py`
  - [ ] Update `_build_state()` method to process `pre_validators` from config
  - [ ] Build pre-validator functions and add to `state_def.pre_validation_functions`
  - [ ] Continue building existing validators into `post_validation_functions`

### 4. Config Loader Updates
- [ ] **File**: `src/dataknobs_fsm/config/loader.py`
  - [ ] Update `_process_state()` to handle `pre_validators` field
  - [ ] Ensure backward compatibility for configs with only `validators`

### 5. State Resource Allocation Implementation

**CRITICAL DESIGN DECISION**: Arc resources must be ADDITIVE to state resources, not replacing them. This prevents resource conflicts and ensures arc functions have access to both state and arc-specific resources.

- [ ] **File**: `src/dataknobs_fsm/execution/engine.py`

  #### Add State Resource Management Methods
  ```python
  def _allocate_state_resources(
      self,
      context: ExecutionContext,
      state_name: str
  ) -> Dict[str, Any]:
      """Allocate resources required by a state.

      Args:
          context: Execution context.
          state_name: Name of the state.

      Returns:
          Dictionary of allocated resources.
      """
      state_def = self.fsm.get_state(state_name)
      if not state_def or not state_def.resource_requirements:
          return {}

      resources = {}
      resource_manager = getattr(context, 'resource_manager', None)
      if not resource_manager:
          return resources

      # Generate owner ID for state resource allocation
      owner_id = f"state_{state_name}_{getattr(context, 'execution_id', 'unknown')}"

      for resource_config in state_def.resource_requirements:
          try:
              resource = resource_manager.acquire(
                  name=resource_config.name,
                  owner_id=owner_id,
                  timeout=resource_config.timeout
              )
              resources[resource_config.name] = resource
          except Exception as e:
              # Log error but continue with other resources
              self._log_error(f"Failed to acquire resource {resource_config.name}: {e}")

      return resources

  def _release_state_resources(
      self,
      context: ExecutionContext,
      state_name: str,
      resources: Dict[str, Any]
  ) -> None:
      """Release state-allocated resources.

      Args:
          context: Execution context.
          state_name: Name of the state.
          resources: Resources to release.
      """
      resource_manager = getattr(context, 'resource_manager', None)
      if not resource_manager:
          return

      owner_id = f"state_{state_name}_{getattr(context, 'execution_id', 'unknown')}"

      for resource_name in resources.keys():
          try:
              resource_manager.release(resource_name, owner_id)
          except Exception as e:
              # Log error but continue
              self._log_error(f"Failed to release resource {resource_name}: {e}")
  ```

  #### Update State Transform Execution
  - [ ] Modify `_execute_state_transforms()` to allocate resources:
    ```python
    def _execute_state_transforms(self, context: ExecutionContext, state_name: str) -> None:
        state_def = self.fsm.get_state(state_name)
        if not state_def:
            return

        # Allocate state resources
        state_resources = self._allocate_state_resources(context, state_name)

        try:
            transform_functions, state_obj = self.prepare_state_transform(state_def, context)

            for transform_func in transform_functions:
                # Create function context WITH resources
                func_context = FunctionContext(
                    state_name=state_name,
                    function_name=getattr(transform_func, '__name__', 'transform'),
                    metadata={'state': state_name},
                    resources=state_resources  # Pass allocated resources
                )

                # Execute transform...

        finally:
            # Release resources when leaving state
            # Note: May need to track this at context level for proper cleanup
            pass
    ```

  #### Add State Resource Tracking to Context
  - [ ] Store current state resources in ExecutionContext
  - [ ] Release previous state's resources when transitioning
  - [ ] Ensure cleanup on error conditions

### 6. Arc Resource Merging Implementation
- [ ] **File**: `src/dataknobs_fsm/core/arc.py`

  #### Update Arc Resource Allocation to Merge with State Resources
  ```python
  def _allocate_resources(
      self,
      context: "ExecutionContext",
      state_resources: Dict[str, Any] = None
  ) -> Dict[str, Any]:
      """Allocate required resources for arc execution, merging with state resources.

      Args:
          context: Execution context.
          state_resources: Already allocated state resources to merge with.

      Returns:
          Dictionary of merged resources (state + arc-specific).
      """
      # Start with state resources if provided
      resources = dict(state_resources) if state_resources else {}

      # Get resource manager from context
      resource_manager = getattr(context, 'resource_manager', None)
      if not resource_manager:
          return resources

      # Generate unique owner ID for this arc execution
      arc_identifier = f"{self.source_state}_to_{self.arc_def.target_state}"
      owner_id = f"arc_{arc_identifier}_{getattr(context, 'execution_id', 'unknown')}"

      for resource_name in self.arc_def.required_resources:
          # Skip if already have this resource from state
          if resource_name in resources:
              self._log_warning(
                  f"Arc resource '{resource_name}' already allocated by state, skipping"
              )
              continue

          try:
              # Acquire arc-specific resource
              resource = resource_manager.acquire(
                  name=resource_name,
                  owner_id=owner_id,
                  timeout=30.0
              )
              resources[resource_name] = resource

              # Track for cleanup (only arc-specific resources)
              if not hasattr(context, '_arc_acquired_resources'):
                  context._arc_acquired_resources = {}
              context._arc_acquired_resources[resource_name] = owner_id

          except Exception as e:
              # Clean up only arc-specific resources, not state resources
              self._release_arc_resources(context, context._arc_acquired_resources)
              raise ResourceError(
                  f"Failed to acquire arc resource '{resource_name}': {e}"
              ) from e

      return resources

  def _release_arc_resources(
      self,
      context: "ExecutionContext",
      arc_resources: Dict[str, str]
  ) -> None:
      """Release only arc-specific resources, not state resources.

      Args:
          context: Execution context.
          arc_resources: Map of resource_name -> owner_id for arc resources only.
      """
      if not arc_resources:
          return

      resource_manager = getattr(context, 'resource_manager', None)
      if not resource_manager:
          return

      for resource_name, owner_id in arc_resources.items():
          try:
              resource_manager.release(resource_name, owner_id)
          except Exception as e:
              self._log_error(f"Failed to release arc resource {resource_name}: {e}")

      # Clear arc resources tracking
      context._arc_acquired_resources = {}
  ```

### 7. Shared Variables Implementation
- [ ] **File**: `src/dataknobs_fsm/functions/base.py`

  #### Extend FunctionContext to Include Variables Access
  ```python
  @dataclass
  class FunctionContext:
      """Context passed to functions during execution."""
      state_name: str
      function_name: str
      metadata: Dict[str, Any] = field(default_factory=dict)
      resources: Dict[str, Any] = field(default_factory=dict)
      variables: Dict[str, Any] = field(default_factory=dict)  # NEW: Shared variables
      network_name: str | None = None  # NEW: Current network for scoping
  ```

- [ ] **File**: `src/dataknobs_fsm/execution/engine.py`

  #### Pass Variables to Function Context
  ```python
  def _create_function_context(
      self,
      context: ExecutionContext,
      state_name: str,
      function_name: str,
      resources: Dict[str, Any] = None
  ) -> FunctionContext:
      """Create function context with access to shared variables.

      Args:
          context: Execution context.
          state_name: Name of current state.
          function_name: Name of function being executed.
          resources: Allocated resources.

      Returns:
          FunctionContext with variables reference.
      """
      return FunctionContext(
          state_name=state_name,
          function_name=function_name,
          metadata={'state': state_name},
          resources=resources or {},
          variables=context.variables,  # Pass reference to shared variables
          network_name=context.current_network  # Include network for scoping
      )
  ```

- [ ] **File**: `src/dataknobs_fsm/execution/network.py`

  #### Variable Scoping for Subnetworks
  ```python
  def _handle_push_arc(
      self,
      arc: PushArc,
      context: ExecutionContext
  ) -> bool:
      """Handle push arc with variable scoping.

      Variables are organized in three scopes:
      - Global: context.variables['_global']
      - Network: context.variables['_networks'][network_name]
      - Local: context.variables['_local'] (network-specific, not inherited)
      """
      # Save current network's local variables
      parent_local_vars = context.variables.get('_local', {}).copy()

      # Push to subnetwork
      context.push_network(arc.target_network, arc.return_state)

      # Initialize subnetwork variable scope
      if '_networks' not in context.variables:
          context.variables['_networks'] = {}
      if arc.target_network not in context.variables['_networks']:
          context.variables['_networks'][arc.target_network] = {}

      # Set local variables for subnetwork (empty initially)
      context.variables['_local'] = {}

      try:
          # Execute subnetwork
          success, result = self.execute_network(
              arc.target_network,
              context,
              context.data
          )

          if success:
              context.data = result

          return success
      finally:
          # Restore parent's local variables
          context.variables['_local'] = parent_local_vars
          context.pop_network()
  ```

  #### Update All Function Context Creation
  - [ ] Update `_execute_state_transforms()` to pass variables
  - [ ] Update `_execute_pre_validators()` to pass variables
  - [ ] Update `_execute_state_functions()` to pass variables
  - [ ] Ensure arc functions also get variables access

- [ ] **File**: `src/dataknobs_fsm/core/arc.py`

  #### Update Arc Function Context Creation
  ```python
  def _create_function_context(
      self,
      exec_context: "ExecutionContext",
      resources: Dict[str, Any] | None = None,
      stream_enabled: bool = False
  ) -> FunctionContext:
      """Create function context for execution.

      Args:
          exec_context: Execution context.
          resources: Allocated resources.
          stream_enabled: Whether streaming is enabled.

      Returns:
          Function context.
      """
      return FunctionContext(
          state_name=self.source_state,
          function_name=self.arc_def.transform or self.arc_def.pre_test,
          metadata={
              'source_state': self.source_state,
              'target_state': self.arc_def.target_state,
              'arc_priority': self.arc_def.priority,
              'stream_enabled': stream_enabled
          },
          resources=resources or {},
          variables=exec_context.variables  # Pass shared variables
      )
  ```

### 8. Variable Scope and Lifecycle Management
- [ ] **File**: `src/dataknobs_fsm/execution/context.py`

  #### Add Variable Management Methods
  ```python
  def set_variable(self, key: str, value: Any, scope: str = 'execution') -> None:
      """Set a shared variable.

      Args:
          key: Variable name.
          value: Variable value.
          scope: Variable scope ('execution', 'network', 'global').
      """
      if scope == 'network':
          # Scope to current network in stack
          scoped_key = f"{self.network_stack[-1][0]}.{key}" if self.network_stack else key
      else:
          scoped_key = key

      self.variables[scoped_key] = value

  def get_variable(self, key: str, default: Any = None, scope: str = 'execution') -> Any:
      """Get a shared variable.

      Args:
          key: Variable name.
          default: Default value if not found.
          scope: Variable scope.

      Returns:
          Variable value or default.
      """
      if scope == 'network':
          scoped_key = f"{self.network_stack[-1][0]}.{key}" if self.network_stack else key
      else:
          scoped_key = key

      return self.variables.get(scoped_key, default)

  def clear_network_variables(self, network_name: str) -> None:
      """Clear variables scoped to a specific network.

      Args:
          network_name: Network name.
      """
      prefix = f"{network_name}."
      keys_to_remove = [k for k in self.variables if k.startswith(prefix)]
      for key in keys_to_remove:
          del self.variables[key]
  ```

### 9. Execution Engine Updates (Synchronous)
- [ ] **File**: `src/dataknobs_fsm/execution/engine.py`

  #### Add Pre-validator Execution Method
  ```python
  def _execute_pre_validators(
      self,
      context: ExecutionContext,
      state_name: str,
      state_resources: Dict[str, Any] = None
  ) -> bool:
      """Execute pre-validation functions when entering a state.

      Args:
          context: Execution context.
          state_name: Name of the state.
          state_resources: Already allocated state resources.

      Returns:
          True if validation passes, False otherwise.
      """
      state_def = self.fsm.get_state(state_name)
      if not state_def:
          return True

      # Use provided resources or empty dict
      resources = state_resources if state_resources is not None else {}

      if hasattr(state_def, 'pre_validation_functions') and state_def.pre_validation_functions:
          for validator_func in state_def.pre_validation_functions:
              try:
                  # Execute validator with state resources
                  func_context = FunctionContext(
                      state_name=state_name,
                      function_name=getattr(validator_func, '__name__', 'validate'),
                      metadata={'state': state_name, 'phase': 'pre_validation'},
                      resources=resources  # Pass state resources
                  )
                  result = validator_func(ensure_dict(context.data), func_context)

                  if result is False:
                      return False
                  # Update context.data if result is a dict
                  if isinstance(result, dict):
                      context.data.update(result)
              except Exception as e:
                  # Log error and fail validation
                  return False
      return True
  ```

  #### Update `_execute_transition()` Method
  - [ ] After `context.set_state(arc.target_state)` (line ~362)
  - [ ] Add integrated state entry processing:
    ```python
    # Allocate state resources once for all state functions
    state_resources = self._allocate_state_resources(context, arc.target_state)

    # Store in context for cleanup tracking
    context.current_state_resources = state_resources

    # Execute pre-validators with state resources
    if not self._execute_pre_validators(context, arc.target_state, state_resources):
        self._release_state_resources(context, arc.target_state, state_resources)
        return False

    # Execute transforms with state resources (modified method)
    self._execute_state_transforms(context, arc.target_state, state_resources)
    ```
  - [ ] Ensure resources are released when leaving state

  #### Update Initial State Entry
  - [ ] In `execute()` method, after setting initial state (line ~102)
  - [ ] Add pre-validator execution before transforms:
    ```python
    if not self._execute_pre_validators(context, initial_state):
        return False, "Pre-validation failed for initial state"
    self._execute_state_transforms(context, initial_state)
    ```

  #### Rename for Clarity
  - [ ] Consider renaming `_execute_state_functions` to `_execute_post_validators`
  - [ ] Update all references

### 6. Async Engine Updates
- [ ] **File**: `src/dataknobs_fsm/execution/async_engine.py`
  - [ ] Add `async def _execute_pre_validators()` method (async version)
  - [ ] Update `_execute_transition()` to call pre-validators
  - [ ] Update initial state entry logic
  - [ ] Consider renaming methods for clarity

### 7. Base Engine Updates
- [ ] **File**: `src/dataknobs_fsm/execution/base_engine.py`
  - [ ] Add pre-validator support to shared base logic if applicable
  - [ ] Update any shared validation helper methods

### 8. SimpleFSM API Updates
- [ ] **File**: `src/dataknobs_fsm/api/simple.py`
- [ ] **File**: `src/dataknobs_fsm/api/async_simple.py`
  - [ ] Ensure SimpleFSM properly handles pre-validators in config
  - [ ] Update any config transformation logic

### 9. AdvancedFSM API Updates
- [ ] **File**: `src/dataknobs_fsm/api/advanced.py`
  - [ ] Support pre-validators in builder pattern
  - [ ] Add `add_pre_validator()` method to state builder
  - [ ] Keep `add_validator()` for post-validators

### 10. Test Updates
- [ ] **Create**: `tests/test_pre_validators.py`
  - [ ] Test pre-validators execute before transforms
  - [ ] Test pre-validator failure prevents transform execution
  - [ ] Test data flow through pre-validators → transforms → post-validators
  - [ ] Test backward compatibility (configs with only `validators`)
  - [ ] Test pre-validators receive state resources

- [ ] **Create**: `tests/test_state_resources.py`
  - [ ] Test state resources are allocated on state entry
  - [ ] Test state functions receive allocated resources
  - [ ] Test resources are released when leaving state
  - [ ] Test resource reuse within same state
  - [ ] Test resource isolation between states

- [ ] **Create**: `tests/test_arc_state_resource_merging.py`
  - [ ] Test arc resources are additive to state resources
  - [ ] Test arc transform receives both state and arc resources
  - [ ] Test duplicate resource names are handled (state takes precedence)
  - [ ] Test arc-only resources are released after arc execution
  - [ ] Test state resources persist after arc execution
  - [ ] Test warning is logged for conflicting resource names

- [ ] **Create**: `tests/test_shared_variables.py`
  - [ ] Test variables are accessible across states
  - [ ] Test variable modifications persist
  - [ ] Test network-scoped variables isolation
  - [ ] Test variables clear when network pops from stack
  - [ ] Test concurrent access safety (if multi-threaded)
  - [ ] Test variable access in validators, transforms, and arc functions

- [ ] **Update Existing Tests**
  - [ ] `tests/test_state.py` - Add pre-validator tests
  - [ ] `tests/test_execution_common.py` - Verify execution order
  - [ ] `tests/test_builder_execution.py` - Test building with pre-validators
  - [ ] `tests/test_resources.py` - Add state resource tests

### 11. Example Updates
- [ ] **Update Examples**
  - [ ] Add pre-validators to at least one example
  - [ ] Add shared variables usage example
  - [ ] Create example showing validation pipeline:
    ```yaml
    states:
      - name: process_data
        pre_validators:
          - type: inline
            code: "lambda state: 'required_field' in state.data"
        transforms:
          - type: inline
            code: "lambda state: {**state.data, 'processed': True}"
        validators:  # post-validators
          - type: inline
            code: "lambda state: state.data.get('processed') == True"
    ```

  - [ ] Create example using shared variables:
    ```python
    # State 1: Load expensive data once
    def load_reference_data(data: Dict[str, Any], context: FunctionContext) -> Dict[str, Any]:
        # Load expensive reference data once
        if 'reference_data' not in context.variables:
            reference_data = load_from_database()  # Expensive operation
            context.variables['reference_data'] = reference_data

        return data

    # State 2: Use cached reference data
    def enrich_with_reference(data: Dict[str, Any], context: FunctionContext) -> Dict[str, Any]:
        # Access previously loaded data
        reference_data = context.variables.get('reference_data', {})

        # Enrich current record
        data['enriched_field'] = reference_data.get(data['key'], 'default')

        return data

    # State 3: Track processing stats
    def update_stats(data: Dict[str, Any], context: FunctionContext) -> Dict[str, Any]:
        # Update shared statistics
        stats = context.variables.get('processing_stats', {'count': 0, 'errors': 0})

        if data.get('success'):
            stats['count'] += 1
        else:
            stats['errors'] += 1

        context.variables['processing_stats'] = stats

        # Optionally include stats in final record
        if context.metadata.get('is_end_state'):
            data['final_stats'] = stats

        return data
    ```

### 12. Documentation Updates
- [ ] **File**: `docs/FSM_PROCESSING_FLOW.md` ✅ (Already updated)
- [ ] **File**: `README.md` - Add note about pre and post validators
- [ ] **File**: `docs/API.md` (if exists) - Document new methods

## Migration Guide

### For Existing Configurations
Existing configurations will continue to work without changes:
- `validators` field continues to work as post-validators
- No breaking changes to existing behavior

### For New Configurations
Users can now optionally add pre-validators:
```yaml
states:
  - name: my_state
    pre_validators:  # NEW - validates incoming data
      - {...}
    transforms:
      - {...}
    validators:  # Existing - validates after transforms
      - {...}
```

## Testing Strategy

### Unit Tests
1. **Pre-validator execution timing**
   - Verify pre-validators run before transforms
   - Verify post-validators run after transforms

2. **Pre-validator failure handling**
   - Pre-validator returns False → transforms should not execute
   - Pre-validator raises exception → transforms should not execute

3. **Data flow**
   - Pre-validator modifies data → transforms see modified data
   - Transforms modify data → post-validators see transformed data

4. **Backward compatibility**
   - Configs with only `validators` work as before
   - Mixed configs (some states with pre-validators, some without)

### Integration Tests
1. **Full execution flow**
   - Create FSM with pre and post validators
   - Execute and verify order of operations
   - Verify data modifications at each stage

2. **Error scenarios**
   - Pre-validator failure in initial state
   - Pre-validator failure during transition
   - Recovery from pre-validation failures

## Task 5: Simplify Arc Selection and Add Fallback Behavior

### Overview
Remove the configurable execution strategy concept and implement deterministic priority-based arc selection with automatic fallback on downstream failures.

### Changes Required

### 1. Remove Execution Strategy Configuration
- [ ] **File**: `src/dataknobs_fsm/config/schema.py`
  - Remove `ExecutionStrategy` enum
  - Remove `execution_strategy` field from `FSMConfig`

- [ ] **File**: `src/dataknobs_fsm/execution/engine.py` and `async_engine.py`
  - Remove `strategy` parameter from constructors
  - Remove `TraversalStrategy` enum
  - Update `_choose_transition` to always use priority-based selection

### 2. Implement Fallback on Downstream Failure
- [ ] **File**: `src/dataknobs_fsm/execution/engine.py`

  #### Update Arc Execution with Fallback
  ```python
  def _execute_single(
      self,
      context: ExecutionContext,
      max_transitions: int,
      arc_name: str | None = None
  ) -> Tuple[bool, Any]:
      """Execute in single record mode with arc fallback.

      Args:
          context: Execution context.
          max_transitions: Maximum transitions.
          arc_name: Optional specific arc name to follow.

      Returns:
          Tuple of (success, result).
      """
      transitions = 0

      while transitions < max_transitions:
          # Check if in final state
          if self._is_final_state(context.current_state):
              return True, context.data

          # Execute state functions
          self._execute_state_functions(context, context.current_state)

          # Get all valid transitions sorted by priority
          transitions_available = self._get_available_transitions(context, arc_name)

          if not transitions_available:
              # No valid transitions
              if self._is_final_state(context.current_state):
                  return True, context.data
              return False, f"No valid transitions from state: {context.current_state}"

          # Try each arc in priority order until one succeeds
          execution_succeeded = False
          last_error = None

          for arc in transitions_available:
              # Create savepoint for rollback
              savepoint = context.create_savepoint()

              try:
                  # Execute transition
                  success = self._execute_transition(context, arc)

                  if success:
                      # Try to continue execution from new state
                      # Recurse to handle downstream execution
                      downstream_success, downstream_result = self._continue_execution(
                          context,
                          max_transitions - transitions - 1
                      )

                      if downstream_success:
                          execution_succeeded = True
                          break
                      else:
                          # Downstream failed, rollback and try next arc
                          context.restore_savepoint(savepoint)
                          last_error = downstream_result
              except Exception as e:
                  # Transition failed, rollback and try next arc
                  context.restore_savepoint(savepoint)
                  last_error = str(e)

          if not execution_succeeded:
              # All arcs failed
              return False, last_error or "All transitions failed"

          transitions += 1

      return False, f"Maximum transitions ({max_transitions}) exceeded"
  ```

### 3. Simplify Transition Selection
- [ ] **File**: `src/dataknobs_fsm/execution/common.py`

  ```python
  def select_transition(
      self,
      available: List[ArcDefinition],
      context: ExecutionContext
  ) -> List[ArcDefinition]:
      """Return all valid transitions in priority order.

      Args:
          available: Available transitions already filtered by pre-tests.
          context: Execution context.

      Returns:
          List of arcs sorted by priority (highest first).
      """
      # Simply return the list as-is since it's already sorted by priority
      # The calling code will try each in order
      return available
  ```

### 4. Add Savepoint Support to ExecutionContext
- [ ] **File**: `src/dataknobs_fsm/core/context.py`

  ```python
  class ExecutionContext:
      """Extended to support savepoints for rollback."""

      def create_savepoint(self) -> Dict[str, Any]:
          """Create a savepoint of current state.

          Returns:
              Savepoint data for restoration.
          """
          return {
              'current_state': self.current_state,
              'data': copy.deepcopy(self.data),
              'variables': copy.deepcopy(self.variables),
              'path': list(self.path),
              'metadata': copy.deepcopy(self.metadata)
          }

      def restore_savepoint(self, savepoint: Dict[str, Any]) -> None:
          """Restore context to a previous savepoint.

          Args:
              savepoint: Previously created savepoint.
          """
          self.current_state = savepoint['current_state']
          self.data = savepoint['data']
          self.variables = savepoint['variables']
          self.path = savepoint['path']
          self.metadata = savepoint['metadata']
  ```

### Expected Benefits
1. **Simpler mental model**: Always priority-based, no strategy configuration to understand
2. **More robust execution**: Automatic fallback to alternative paths
3. **Better error recovery**: Can try multiple paths before failing
4. **Cleaner API**: Remove unnecessary configuration options

### Risk Assessment for Arc Selection Changes
- **Low Risk**: Simplifies existing behavior
- **Breaking Change**: Removes `execution_strategy` config (but it wasn't widely used)
- **Migration**: Existing FSMs will work better with automatic fallback

## Task 6: Fix Arc Definition Order Preservation

### Overview
Ensure that arcs with equal priority are selected in their definition order. Currently, the sort operation doesn't preserve definition order for arcs with equal priority.

### Changes Required

### 1. Add Definition Order Tracking
- [ ] **File**: `src/dataknobs_fsm/core/arc.py`

  ```python
  @dataclass
  class ArcDefinition:
      """Definition of an arc between states."""

      target_state: str
      pre_test: str | None = None
      transform: str | None = None
      priority: int = 0  # Higher priority arcs are evaluated first
      definition_order: int = 0  # NEW: Track definition order for stable sorting
      resources: List[str] = field(default_factory=list)
      metadata: Dict[str, Any] = field(default_factory=dict)
  ```

### 2. Set Definition Order When Building FSM
- [ ] **File**: `src/dataknobs_fsm/config/builder.py`

  ```python
  def _build_network(self, network_config: NetworkConfig) -> StateNetwork:
      """Build a state network from configuration."""
      # ... existing code ...

      # Track arc definition order globally across all states
      arc_definition_order = 0

      for state_config in network_config.states:
          state = State(name=state_config.name, ...)

          # Add arcs with definition order
          for arc_config in state_config.arcs:
              arc_def = ArcDefinition(
                  target_state=arc_config.target,
                  pre_test=arc_config.pre_test,
                  transform=arc_config.transform,
                  priority=arc_config.priority,
                  definition_order=arc_definition_order,  # NEW
                  resources=arc_config.resources,
                  metadata=arc_config.metadata
              )
              arc_definition_order += 1
              # ... rest of arc creation ...
  ```

### 3. Update Arc Sorting to Use Stable Sort
- [ ] **File**: `src/dataknobs_fsm/execution/engine.py`

  ```python
  def _get_available_transitions(
      self,
      context: ExecutionContext,
      arc_name: str | None = None
  ) -> List[ArcDefinition]:
      """Get available transitions from current state."""
      # ... existing code to collect available arcs ...

      # Sort by priority (descending) then by definition order (ascending)
      # This ensures stable ordering when priorities are equal
      available.sort(
          key=lambda x: (-x.priority, x.definition_order)
      )

      return available
  ```

### 4. Update Network Arc Conversion
- [ ] **File**: `src/dataknobs_fsm/core/network.py`

  ```python
  @property
  def arcs(self) -> Dict[str, Any]:
      """Get all arcs in the network."""
      from dataknobs_fsm.core.arc import ArcDefinition

      arc_dict = {}
      for idx, arc in enumerate(self._arcs):
          key = f"{arc.source_state}:{arc.target_state}"
          arc_def = ArcDefinition(
              target_state=arc.target_state,
              pre_test=arc.pre_test,
              transform=arc.transform,
              priority=getattr(arc, 'priority', 0),
              definition_order=idx,  # Use list index as definition order
              resources=getattr(arc, 'resources', []),
              metadata=getattr(arc, 'metadata', {})
          )
          arc_dict[key] = arc_def
      return arc_dict
  ```

### Expected Benefits
1. **Predictable behavior**: Arcs are always evaluated in a consistent order
2. **Backward compatible**: Existing FSMs work without changes
3. **Clear precedence**: Definition order provides natural tiebreaker

### Risk Assessment for Definition Order Fix
- **Low Risk**: Only affects arc selection when priorities are equal
- **No Breaking Changes**: Adds field but maintains compatibility
- **Improves Predictability**: Makes behavior more deterministic

## Risk Assessment

### Low Risk
- Additive change (new feature, not modifying existing)
- Backward compatible
- Clear separation of concerns

### Medium Risk
- Execution order complexity increases
- More validation points could impact performance
- Need clear documentation to avoid confusion

### Mitigation
- Comprehensive testing
- Clear naming (pre_validators vs validators)
- Performance benchmarks before/after
- Gradual rollout with feature flag if needed

## Implementation Order

1. **Phase 1: Core Implementation**
   - Schema updates
   - State definition updates
   - Basic execution engine updates

2. **Phase 2: Engine Integration**
   - Complete sync engine implementation
   - Complete async engine implementation
   - Add comprehensive tests

3. **Phase 3: API Updates**
   - Update SimpleFSM/AdvancedFSM
   - Update builders and loaders
   - Update examples

4. **Phase 4: Documentation & Release**
   - Update all documentation
   - Create migration guide
   - Performance testing
   - Release notes

## Estimated Effort

- **Core Implementation**: 4-6 hours
- **Testing**: 3-4 hours
- **Documentation**: 1-2 hours
- **Total**: 8-12 hours

## Success Criteria

1. Pre-validators execute before transforms in all execution modes
2. Pre-validator failure prevents transform execution
3. State resources are allocated when entering a state
4. All state functions receive their state's allocated resources
5. Arc resources are properly merged with state resources (additive)
6. Resources are properly released when leaving a state
7. Shared variables are accessible across all states in an execution
8. Variables can be scoped to networks and cleared appropriately
9. Functions can read and write to shared variables dictionary
10. Arc selection is always deterministic priority-based
11. Arcs with equal priority are selected in definition order
12. Failed downstream execution triggers fallback to next priority arc
13. Savepoints enable rollback on arc failure
14. Execution strategy configuration is removed
15. All existing tests pass (backward compatibility)
16. New tests pass for all new functionality
17. Documentation clearly explains validator types, resource scopes, and variable usage
18. Examples demonstrate all new features working together
19. No performance regression for FSMs without these features
20. Thread-safe variable access (if concurrent execution supported)

## Notes

- Consider deprecation warning for ambiguous use of "validators" alone
- Future: Consider adding "validator_mode" config to make behavior explicit
- Could add syntactic sugar like `@pre_validator` and `@post_validator` decorators
