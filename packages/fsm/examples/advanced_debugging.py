#!/usr/bin/env python3
"""
Advanced FSM Debugging and Monitoring Example.

This example demonstrates:
1. Step-by-step execution with breakpoints
2. Execution tracing and profiling
3. Custom execution hooks for monitoring
4. State inspection and debugging
5. Performance profiling
6. Error handling with detailed diagnostics
"""

import time
import json
from typing import Dict, Any, List
from datetime import datetime
from dataknobs_fsm.api.advanced import (
    AdvancedFSM,
    ExecutionMode,
    ExecutionHook,
    FSMDebugger,
    create_advanced_fsm
)
from dataknobs_fsm.core.data_modes import DataHandlingMode


# Custom functions for the workflow
def validate_input(state) -> Dict[str, Any]:
    """Validate input data."""
    data = state.data.copy()

    # Simulate validation
    required_fields = ['user_id', 'request_type', 'payload']
    missing = [f for f in required_fields if f not in data]

    if missing:
        data['validation_errors'] = f"Missing fields: {missing}"
        data['is_valid'] = False
    else:
        data['is_valid'] = True
        data['validated_at'] = datetime.now().isoformat()

    # Simulate some processing time
    time.sleep(0.01)  # Reduced for faster demo

    return data


def process_request(state) -> Dict[str, Any]:
    """Process the validated request."""
    data = state.data.copy()
    
    request_type = data.get('request_type', 'unknown')
    
    # Simulate different processing based on request type
    if request_type == 'compute':
        # Simulate computation
        result = sum(range(1000))
        data['result'] = result
        time.sleep(0.02)  # Simulate compute time
    elif request_type == 'query':
        # Simulate database query
        data['result'] = {'records': 42, 'status': 'complete'}
        time.sleep(0.015)
    else:
        data['result'] = {'status': 'unknown_type'}
        time.sleep(0.005)
    
    data['processed_at'] = datetime.now().isoformat()
    data['processing_complete'] = True
    
    return data


def enrich_data(state) -> Dict[str, Any]:
    """Enrich processed data with additional information."""
    data = state.data.copy()
    
    # Add metadata
    data['metadata'] = {
        'version': '1.0',
        'processor': 'advanced_fsm',
        'enrichment_timestamp': datetime.now().isoformat()
    }
    
    # Add computed fields
    if 'result' in data:
        data['has_result'] = True
        data['result_type'] = type(data['result']).__name__
    
    time.sleep(0.005)
    
    return data


def format_output(state) -> Dict[str, Any]:
    """Format the final output."""
    data = state.data.copy()
    
    # Create formatted response
    data['formatted_response'] = {
        'request_id': data.get('request_id', 'unknown'),
        'user_id': data.get('user_id'),
        'status': 'success' if data.get('processing_complete') else 'incomplete',
        'result': data.get('result'),
        'metadata': data.get('metadata'),
        'timestamps': {
            'validated': data.get('validated_at'),
            'processed': data.get('processed_at'),
            'formatted': datetime.now().isoformat()
        }
    }
    
    time.sleep(0.005)
    
    return data


def check_validation(data: Dict[str, Any], context: Any) -> bool:
    """Check if validation passed."""
    return data.get('is_valid', False)


def check_processing(data: Dict[str, Any], context: Any) -> bool:
    """Check if processing completed."""
    return data.get('processing_complete', False)


# FSM configuration
debug_workflow_config = {
    "name": "DebugWorkflow",
    "main_network": "main",
    "networks": [{
        "name": "main",
        "states": [
            {
                "name": "start",
                "is_start": True
            },
            {
                "name": "validate",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "validate_input"
                    }
                }
            },
            {
                "name": "process",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "process_request"
                    }
                }
            },
            {
                "name": "enrich",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "enrich_data"
                    }
                }
            },
            {
                "name": "format",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "format_output"
                    }
                }
            },
            {
                "name": "success",
                "is_end": True
            },
            {
                "name": "validation_failed",
                "is_end": True
            },
            {
                "name": "processing_failed",
                "is_end": True
            }
        ],
        "arcs": [
            {"from": "start", "to": "validate"},
            {
                "from": "validate",
                "to": "process",
                "condition": {
                    "type": "registered",
                    "name": "check_validation"
                }
            },
            {
                "from": "validate",
                "to": "validation_failed",
                "condition": {
                    "type": "inline",
                    "code": "not data.get('is_valid', False)"
                }
            },
            {
                "from": "process",
                "to": "enrich",
                "condition": {
                    "type": "registered",
                    "name": "check_processing"
                }
            },
            {
                "from": "process",
                "to": "processing_failed",
                "condition": {
                    "type": "inline",
                    "code": "not data.get('processing_complete', False)"
                }
            },
            {"from": "enrich", "to": "format"},
            {"from": "format", "to": "success"}
        ]
    }]
}


def demonstrate_step_by_step_execution():
    """Demonstrate step-by-step execution with inspection."""
    print("\n" + "=" * 70)
    print("📍 STEP-BY-STEP EXECUTION (SYNCHRONOUS)")
    print("=" * 70)

    # Create advanced FSM with custom functions
    fsm = create_advanced_fsm(
        debug_workflow_config,
        custom_functions={
            'validate_input': validate_input,
            'process_request': process_request,
            'enrich_data': enrich_data,
            'format_output': format_output,
            'check_validation': check_validation,
            'check_processing': check_processing
        },
        execution_mode=ExecutionMode.STEP_BY_STEP
    )

    # Initialize with test data
    test_data = {
        'request_id': 'REQ-001',
        'user_id': 'USER-123',
        'request_type': 'compute',
        'payload': {'data': [1, 2, 3, 4, 5]}
    }

    print(f"\nStarting with data: {json.dumps(test_data, indent=2)}")

    # Create synchronous context
    context = fsm.create_context(test_data)
    steps = []

    while not context.is_complete():
        # Get current state
        current = context.get_current_state()
        print(f"\n📍 Current State: {current}")

        # Get available transitions from current state
        transitions = fsm.get_available_transitions(current)
        if transitions:
            print(f"   Available transitions: {[t['name'] for t in transitions]}")

        # Execute one step synchronously
        step_result = fsm.execute_step_sync(context)
        steps.append(step_result)

        # Show what happened
        print(f"   ✓ Transition: {step_result.transition}")
        print(f"   → To State: {step_result.to_state}")
        print(f"   Duration: {step_result.duration:.3f}s")

        # Show if we hit a breakpoint
        if step_result.at_breakpoint:
            print(f"   🔴 At breakpoint!")

        # Show data snapshot
        data_snapshot = context.get_data_snapshot()
        if 'is_valid' in data_snapshot:
            print(f"   Data: is_valid={data_snapshot['is_valid']}")
        if 'processing_complete' in data_snapshot:
            print(f"   Data: processing_complete={data_snapshot['processing_complete']}")
        if 'validation_errors' in data_snapshot:
            print(f"   Data: validation_errors={data_snapshot['validation_errors']}")

        # Allow inspection (in real use, could pause here)
        # input("Press Enter to continue...")

    print(f"\n✅ Execution complete!")
    print(f"Final State: {context.get_current_state()}")
    print(f"Total Steps: {len(steps)}")


def demonstrate_execution_with_breakpoints():
    """Demonstrate execution with breakpoints."""
    print("\n" + "=" * 70)
    print("🔴 EXECUTION WITH BREAKPOINTS (SYNCHRONOUS)")
    print("=" * 70)

    # Create FSM with breakpoint mode
    fsm = create_advanced_fsm(
        debug_workflow_config,
        custom_functions={
            'validate_input': validate_input,
            'process_request': process_request,
            'enrich_data': enrich_data,
            'format_output': format_output,
            'check_validation': check_validation,
            'check_processing': check_processing
        },
        execution_mode=ExecutionMode.BREAKPOINT
    )

    # Set breakpoints using the API
    fsm.add_breakpoint('process')
    fsm.add_breakpoint('format')

    print(f"\nBreakpoints set at: {fsm.breakpoints}")

    test_data = {
        'request_id': 'REQ-002',
        'user_id': 'USER-456',
        'request_type': 'query',
        'payload': {'query': 'SELECT * FROM users'}
    }

    # Execute with breakpoints synchronously
    print(f"\nExecuting with breakpoints...")

    context = fsm.create_context(test_data)
    path = []

    # Track initial state
    current_state = context.get_current_state()
    if current_state:
        path.append(current_state)

    while not context.is_complete():
        # Run until we hit a breakpoint
        state = fsm.run_until_breakpoint_sync(context)

        if state:
            current = state.definition.name

            # Check if we're at a breakpoint
            if current in fsm.breakpoints:
                print(f"\n🔴 BREAKPOINT HIT at '{current}'")
                data_snapshot = context.get_data_snapshot()
                print(f"   Data (truncated): {json.dumps(data_snapshot, indent=2)[:200]}...")

                # Inspect the state
                state_info = fsm.inspect_state(current)
                print(f"   Has transform: {state_info.get('has_transform', False)}")

                # Step once to move past the breakpoint
                step_result = fsm.execute_step_sync(context)
                if step_result.to_state != current:
                    path.append(step_result.to_state)
        else:
            # No breakpoint hit, execution must be complete
            final_state = context.get_current_state()
            if final_state and final_state not in path:
                path.append(final_state)
            break

    print(f"\nExecution completed!")
    print(f"Path taken: {' -> '.join(path)}")


def demonstrate_execution_tracing():
    """Demonstrate execution tracing and profiling."""
    print("\n" + "=" * 70)
    print("📊 EXECUTION TRACING & PROFILING (SYNCHRONOUS)")
    print("=" * 70)

    # Create FSM
    fsm = create_advanced_fsm(
        debug_workflow_config,
        custom_functions={
            'validate_input': validate_input,
            'process_request': process_request,
            'enrich_data': enrich_data,
            'format_output': format_output,
            'check_validation': check_validation,
            'check_processing': check_processing
        }
    )

    # Enable history tracking for detailed tracing
    fsm.enable_history(max_depth=100)

    test_data = {
        'request_id': 'REQ-003',
        'user_id': 'USER-789',
        'request_type': 'compute',
        'payload': {'numbers': list(range(100))}
    }

    print(f"\nExecuting with tracing enabled...")

    # Execute with tracing synchronously
    start_time = time.time()
    trace = fsm.trace_execution_sync(test_data)
    end_time = time.time()

    print(f"\n✅ Execution completed in {end_time - start_time:.3f} seconds")

    # Show trace results
    print(f"\n📊 Execution Trace:")
    for i, event in enumerate(trace[:10], 1):  # Show first 10 events
        print(f"  {i}. {event['from_state']} -> {event['to_state']}")
        if 'duration' in event:
            print(f"     Duration: {event['duration']:.3f}s")

    # Execute with profiling synchronously
    print(f"\n⏱️ Profiling execution...")
    profile = fsm.profile_execution_sync(test_data)

    print(f"\nPerformance Profile:")
    print(f"  Total time: {profile.get('total_time', 0):.3f}s")
    print(f"  Total transitions: {profile.get('transitions', 0)}")

    if 'state_times' in profile:
        print(f"\n  State execution times:")
        for state, timing in profile['state_times'].items():
            if isinstance(timing, dict):
                # Handle dict format from profile
                duration = timing.get('duration', 0)
                print(f"    • {state}: {duration:.3f}s")
            else:
                # Handle direct numeric value
                print(f"    • {state}: {timing:.3f}s")


def demonstrate_execution_hooks():
    """Demonstrate execution hooks for monitoring."""
    print("\n" + "=" * 70)
    print("🪝 EXECUTION HOOKS")
    print("=" * 70)
    
    # Create monitoring hooks
    state_times = {}
    errors = []
    
    def on_state_enter(state, data):
        state_times[state] = time.time()
        print(f"  → Entering state: {state}")
    
    def on_state_exit(state, data):
        if state in state_times:
            duration = time.time() - state_times[state]
            print(f"  ← Exiting state: {state} (took {duration:.3f}s)")
    
    def on_error(error, state, data):
        errors.append({'error': str(error), 'state': state})
        print(f"  ⚠️ Error in state {state}: {error}")
    
    # Create hooks
    hooks = ExecutionHook(
        on_state_enter=on_state_enter,
        on_state_exit=on_state_exit,
        on_error=on_error
    )
    
    # Create FSM with hooks
    fsm = create_advanced_fsm(
        debug_workflow_config,
        custom_functions={
            'validate_input': validate_input,
            'process_request': process_request,
            'enrich_data': enrich_data,
            'format_output': format_output,
            'check_validation': check_validation,
            'check_processing': check_processing
        }
    )
    
    # Set hooks
    fsm.set_hooks(hooks)
    
    test_data = {
        'request_id': 'REQ-004',
        'user_id': 'USER-999',
        'request_type': 'compute',
        'payload': {'compute': 'heavy'}
    }
    
    print(f"\nExecuting with hooks...")
    # Use trace_execution_sync which will trigger hooks
    result = fsm.trace_execution_sync(test_data)

    print(f"\n✅ Execution completed!")
    print(f"  Transitions executed: {len(result)}")
    if errors:
        print(f"⚠️ Errors encountered: {errors}")


def demonstrate_fsm_debugger():
    """Demonstrate interactive FSM debugger."""
    print("\n" + "=" * 70)
    print("🐛 FSM DEBUGGER (SYNCHRONOUS API)")
    print("=" * 70)

    # Create FSM
    fsm = create_advanced_fsm(
        debug_workflow_config,
        custom_functions={
            'validate_input': validate_input,
            'process_request': process_request,
            'enrich_data': enrich_data,
            'format_output': format_output,
            'check_validation': check_validation,
            'check_processing': check_processing
        }
    )

    # Enable history for debugging
    fsm.enable_history(max_depth=50)

    # Create debugger with the FSMDebugger class
    debugger = FSMDebugger(fsm)

    test_data = {
        'request_id': 'REQ-005',
        'user_id': 'USER-111',
        'request_type': 'query',
        'payload': {'sql': 'SELECT COUNT(*) FROM logs'}
    }

    print("\n🐛 Starting FSM Debugger")
    print("Using the synchronous FSMDebugger class")

    # Start the debug session
    debugger.start(test_data)
    print(f"\nDebug session started")
    print(f"Initial state: {debugger.current_state}")

    # Set breakpoints
    fsm.add_breakpoint('process')
    fsm.add_breakpoint('format')
    print(f"Breakpoints: {fsm.breakpoints}")

    # Add watches
    debugger.watch('validation', 'is_valid')
    debugger.watch('processing', 'processing_complete')

    # Step through execution
    print("\n📍 Stepping through execution:")

    step_count = 0
    while not debugger.context.is_complete() and step_count < 10:
        step_count += 1

        # Execute a single step
        result = debugger.step()

        print(f"\nStep {debugger.step_count}:")
        print(f"  {result.from_state} -> {result.to_state}")
        print(f"  Transition: {result.transition}")
        print(f"  Duration: {result.duration:.3f}s")

        # Show watches
        if debugger.watches:
            print(f"  Watches:")
            for name, value in debugger.watches.items():
                print(f"    {name}: {value}")

        # If at breakpoint, show detailed info
        if result.at_breakpoint:
            print(f"\n  🔴 BREAKPOINT HIT at '{result.to_state}'")
            state_info = debugger.inspect_current_state()
            print(f"  State info: {state_info}")

            # Continue to next breakpoint
            print(f"\n  Continuing to next breakpoint...")
            next_state = debugger.continue_to_breakpoint()
            if next_state:
                print(f"  Stopped at: {next_state.definition.name}")

        if result.is_complete:
            print(f"\n✅ Execution complete!")
            break

    # Show execution history
    print("\n📜 Execution History:")
    history = debugger.get_history(limit=5)
    for i, step in enumerate(history, 1):
        print(f"  {i}. {step.from_state} -> {step.to_state} ({step.transition})")

    print("\n✅ Debug session complete!")
    print(f"Total steps: {debugger.step_count}")
    print(f"Final state: {debugger.current_state}")


def main():
    """Run all advanced FSM demonstrations."""
    print("Advanced FSM Debugging and Monitoring Example")
    print("=" * 70)
    print("This example demonstrates advanced FSM features:")
    print("- Step-by-step execution with inspection")
    print("- Breakpoint debugging")
    print("- Execution tracing and profiling")
    print("- Custom execution hooks")
    print("- Interactive debugging")
    
    # Run all demonstrations
    demonstrate_step_by_step_execution()
    
    # Run automated demonstrations
    demonstrate_execution_with_breakpoints()
    demonstrate_execution_tracing()
    demonstrate_execution_hooks()
    demonstrate_fsm_debugger()
    
    print("\n" + "=" * 70)
    print("Advanced FSM Example Complete!")
    print("\n📌 Key Features Demonstrated:")
    print("  • Step-by-step execution for debugging")
    print("  • Breakpoint support for state inspection")
    print("  • Execution tracing for analysis")
    print("  • Performance profiling")
    print("  • Custom hooks for monitoring")
    print("  • Interactive debugging capabilities")


if __name__ == "__main__":
    main()
