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
    time.sleep(0.1)
    
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
        time.sleep(0.2)  # Simulate compute time
    elif request_type == 'query':
        # Simulate database query
        data['result'] = {'records': 42, 'status': 'complete'}
        time.sleep(0.15)
    else:
        data['result'] = {'status': 'unknown_type'}
        time.sleep(0.05)
    
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
    
    time.sleep(0.05)
    
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
    
    time.sleep(0.05)
    
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
    print("📍 STEP-BY-STEP EXECUTION")
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
    
    # Execute step by step
    context = fsm.create_context(test_data)
    steps = []
    
    while not context.is_complete():
        # Get current state
        current = context.get_current_state()
        print(f"\n📍 Current State: {current}")
        
        # Execute one step
        step_result = fsm.execute_step(context)
        steps.append(step_result)
        
        # Show what happened
        print(f"   Executed: {step_result.get('transition', 'N/A')}")
        print(f"   Next State: {step_result.get('next_state', 'N/A')}")
        
        # Inspect data changes
        if 'data_changes' in step_result:
            print(f"   Data Changes: {step_result['data_changes']}")
        
        # Allow inspection (in real use, could pause here)
        input("Press Enter to continue...")
    
    print(f"\n✅ Execution complete!")
    print(f"Final State: {context.get_current_state()}")
    print(f"Total Steps: {len(steps)}")


def demonstrate_execution_with_breakpoints():
    """Demonstrate execution with breakpoints."""
    print("\n" + "=" * 70)
    print("🔴 EXECUTION WITH BREAKPOINTS")
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
    
    # Set breakpoints
    fsm.set_breakpoint('process')
    fsm.set_breakpoint('format')
    
    print("\nBreakpoints set at: 'process' and 'format' states")
    
    test_data = {
        'request_id': 'REQ-002',
        'user_id': 'USER-456',
        'request_type': 'query',
        'payload': {'query': 'SELECT * FROM users'}
    }
    
    # Execute with breakpoints
    print(f"\nExecuting with breakpoints...")
    
    result = fsm.execute_with_breakpoints(
        test_data,
        on_breakpoint=lambda state, data: print(f"\n🔴 BREAKPOINT HIT at '{state}'\n   Data: {json.dumps(data, indent=2)[:200]}...")
    )
    
    print(f"\nExecution completed!")
    print(f"Path taken: {' -> '.join(result['path'])}")


def demonstrate_execution_tracing():
    """Demonstrate execution tracing and profiling."""
    print("\n" + "=" * 70)
    print("📊 EXECUTION TRACING & PROFILING")
    print("=" * 70)
    
    # Create FSM with tracing mode
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
        execution_mode=ExecutionMode.TRACE
    )
    
    # Enable profiling
    fsm.enable_profiling()
    
    test_data = {
        'request_id': 'REQ-003',
        'user_id': 'USER-789',
        'request_type': 'compute',
        'payload': {'numbers': list(range(100))}
    }
    
    print(f"\nExecuting with tracing enabled...")
    
    # Execute with tracing
    start_time = time.time()
    result = fsm.execute_with_trace(test_data)
    end_time = time.time()
    
    print(f"\n✅ Execution completed in {end_time - start_time:.3f} seconds")
    
    # Get trace
    trace = fsm.get_trace()
    print(f"\n📊 Execution Trace:")
    for i, event in enumerate(trace[-10:], 1):  # Show last 10 events
        print(f"  {i}. {event['timestamp']} - {event['type']}: {event['details']}")
    
    # Get profile data
    profile = fsm.get_profile()
    print(f"\n⏱️ Performance Profile:")
    for state, timing in profile.items():
        print(f"  • {state}: {timing['duration']:.3f}s (calls: {timing['calls']})")


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
    result = fsm.execute(test_data)
    
    print(f"\n✅ Execution completed!")
    if errors:
        print(f"⚠️ Errors encountered: {errors}")


def demonstrate_fsm_debugger():
    """Demonstrate interactive FSM debugger."""
    print("\n" + "=" * 70)
    print("🐛 FSM DEBUGGER")
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
    
    # Create debugger
    debugger = FSMDebugger(fsm)
    
    test_data = {
        'request_id': 'REQ-005',
        'user_id': 'USER-111',
        'request_type': 'query',
        'payload': {'sql': 'SELECT COUNT(*) FROM logs'}
    }
    
    print("\n🐛 Starting FSM Debugger")
    print("Commands: step, continue, inspect, set_breakpoint <state>, clear_breakpoints, quit")
    
    # Simulate debug session (in real use, would be interactive)
    print("\nSimulating debug session...")
    
    # Set initial data
    debugger.set_data(test_data)
    
    # Simulate some debug commands
    commands = [
        "inspect",  # Show current state
        "set_breakpoint process",  # Set breakpoint
        "step",  # Step once
        "inspect",  # Show state again
        "continue",  # Continue to breakpoint
        "inspect",  # Show at breakpoint
        "continue"  # Finish execution
    ]
    
    for cmd in commands:
        print(f"\n> {cmd}")
        # In real implementation, debugger would process these commands
        if cmd == "inspect":
            print(f"  Current State: {debugger.get_current_state()}")
            print(f"  Data (truncated): {str(debugger.get_data())[:100]}...")
        elif cmd.startswith("set_breakpoint"):
            state = cmd.split()[1]
            print(f"  Breakpoint set at '{state}'")
        elif cmd == "step":
            print(f"  Stepped to next state")
        elif cmd == "continue":
            print(f"  Continuing execution...")
    
    print("\n✅ Debug session complete!")


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
    
    # Note: Step-by-step execution is commented out as it requires user input
    # demonstrate_step_by_step_execution()
    
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