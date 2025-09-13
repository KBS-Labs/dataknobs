#!/usr/bin/env python3
"""
Simple Advanced FSM Example showcasing actual AdvancedFSM features.

This example demonstrates:
1. Using AdvancedFSM vs SimpleFSM
2. Setting breakpoints
3. Inspecting states
4. Getting available transitions
5. FSM visualization
"""

import asyncio
import json
from typing import Dict, Any
from datetime import datetime
from dataknobs_fsm.api.advanced import (
    AdvancedFSM,
    ExecutionMode,
    ExecutionHook,
    create_advanced_fsm
)


# Custom functions for the workflow
def initialize_workflow(state) -> Dict[str, Any]:
    """Initialize the workflow."""
    data = state.data.copy()
    data['workflow_id'] = f"WF-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    data['initialized'] = True
    data['steps_completed'] = []
    print(f"  ✓ Initialized workflow: {data['workflow_id']}")
    return data


def validate_input(state) -> Dict[str, Any]:
    """Validate input data."""
    data = state.data.copy()
    
    required = ['user_id', 'action']
    missing = [f for f in required if f not in data]
    
    if missing:
        data['is_valid'] = False
        data['validation_errors'] = f"Missing: {missing}"
        print(f"  ✗ Validation failed: {data['validation_errors']}")
    else:
        data['is_valid'] = True
        print(f"  ✓ Validation passed")
    
    data['steps_completed'].append('validate')
    return data


def process_action(state) -> Dict[str, Any]:
    """Process the action."""
    data = state.data.copy()
    
    action = data.get('action', 'unknown')
    print(f"  → Processing action: {action}")
    
    if action == 'create':
        data['result'] = {'id': 123, 'status': 'created'}
    elif action == 'update':
        data['result'] = {'id': data.get('id', 0), 'status': 'updated'}
    elif action == 'delete':
        data['result'] = {'id': data.get('id', 0), 'status': 'deleted'}
    else:
        data['result'] = {'error': 'Unknown action'}
    
    data['processed'] = True
    data['steps_completed'].append('process')
    print(f"  ✓ Action processed: {data['result']}")
    return data


def finalize_workflow(state) -> Dict[str, Any]:
    """Finalize the workflow."""
    data = state.data.copy()
    
    data['completed_at'] = datetime.now().isoformat()
    data['status'] = 'completed'
    data['steps_completed'].append('finalize')
    
    print(f"  ✓ Workflow finalized at {data['completed_at']}")
    return data


def check_validation(data: Dict[str, Any], context: Any) -> bool:
    """Check if validation passed."""
    return data.get('is_valid', False)


# FSM configuration
workflow_config = {
    "name": "SimpleWorkflow",
    "main_network": "main",
    "networks": [{
        "name": "main",
        "states": [
            {
                "name": "start",
                "is_start": True
            },
            {
                "name": "initialize",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "initialize_workflow"
                    }
                }
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
                        "name": "process_action"
                    }
                }
            },
            {
                "name": "finalize",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "finalize_workflow"
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
            }
        ],
        "arcs": [
            {"from": "start", "to": "initialize"},
            {"from": "initialize", "to": "validate"},
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
            {"from": "process", "to": "finalize"},
            {"from": "finalize", "to": "success"}
        ]
    }]
}


def demonstrate_advanced_features():
    """Demonstrate AdvancedFSM features."""
    print("\n" + "=" * 70)
    print("Advanced FSM Features Demonstration")
    print("=" * 70)
    
    # Create AdvancedFSM with custom functions
    custom_functions = {
        'initialize_workflow': initialize_workflow,
        'validate_input': validate_input,
        'process_action': process_action,
        'finalize_workflow': finalize_workflow,
        'check_validation': check_validation
    }
    
    # Create the advanced FSM
    fsm = create_advanced_fsm(
        workflow_config,
        custom_functions=custom_functions,
        execution_mode=ExecutionMode.STEP_BY_STEP
    )
    
    print("\n1️⃣ FSM Created with AdvancedFSM")
    print(f"   Mode: {fsm.execution_mode}")
    
    # Add breakpoints
    print("\n2️⃣ Setting Breakpoints")
    fsm.add_breakpoint('validate')
    fsm.add_breakpoint('finalize')
    print(f"   Breakpoints set at: validate, finalize")
    
    # Inspect states
    print("\n3️⃣ Inspecting States")
    for state_name in ['start', 'validate', 'process']:
        info = fsm.inspect_state(state_name)
        print(f"   {state_name}:")
        print(f"     - Is Start: {info.get('is_start', False)}")
        print(f"     - Is End: {info.get('is_end', False)}")
        print(f"     - Has Transform: {info.get('has_transform', False)}")
    
    # Get available transitions
    print("\n4️⃣ Available Transitions")
    for state_name in ['start', 'validate']:
        transitions = fsm.get_available_transitions(state_name)
        print(f"   From '{state_name}':")
        for trans in transitions:
            print(f"     → {trans['target']} (has_condition: {trans.get('has_pre_test', False)})")
    
    # Visualize FSM structure
    print("\n5️⃣ FSM Structure Visualization")
    viz = fsm.visualize_fsm()
    print(viz)
    
    # Set up execution hooks
    print("\n6️⃣ Setting Execution Hooks")
    
    def on_state_enter(state):
        print(f"   [HOOK] Entering state: {state}")
    
    def on_state_exit(state):
        print(f"   [HOOK] Exiting state: {state}")
    
    def on_error(error, state, data):
        print(f"   [HOOK] Error in {state}: {error}")
    
    hooks = ExecutionHook(
        on_state_enter=on_state_enter,
        on_state_exit=on_state_exit,
        on_error=on_error
    )
    fsm.set_hooks(hooks)
    print("   Hooks configured for state enter/exit and errors")
    
    # Enable history tracking
    print("\n7️⃣ Enabling History Tracking")
    fsm.enable_history(max_depth=50)
    print("   History tracking enabled (max depth: 50)")
    
    print("\n" + "=" * 70)
    print("Feature demonstration complete!")
    print("Note: Actual execution would use async methods like:")
    print("  - fsm.step(context) for step-by-step execution")
    print("  - fsm.run_until_breakpoint(context) for breakpoint debugging")
    print("  - fsm.trace_execution(data) for full tracing")
    print("  - fsm.profile_execution(data) for performance profiling")


async def demonstrate_async_execution():
    """Demonstrate async execution with AdvancedFSM."""
    print("\n" + "=" * 70)
    print("Async Execution Demonstration")
    print("=" * 70)
    
    # Create AdvancedFSM
    custom_functions = {
        'initialize_workflow': initialize_workflow,
        'validate_input': validate_input,
        'process_action': process_action,
        'finalize_workflow': finalize_workflow,
        'check_validation': check_validation
    }
    
    fsm = create_advanced_fsm(
        workflow_config,
        custom_functions=custom_functions,
        execution_mode=ExecutionMode.TRACE
    )
    
    # Test data
    test_data = {
        'user_id': 'USER-123',
        'action': 'create',
        'data': {'name': 'Test Item'}
    }
    
    print(f"\nInput Data: {json.dumps(test_data, indent=2)}")
    
    # Execute with tracing
    print("\n🔍 Executing with Trace Mode...")
    trace = await fsm.trace_execution(test_data)
    
    print("\n📊 Execution Trace:")
    for entry in trace:
        print(f"  {entry['from']} → {entry['to']}")
    
    # Execute with profiling
    print("\n⏱️ Executing with Profiling...")
    profile = await fsm.profile_execution(test_data)
    
    print("\n📈 Performance Profile:")
    print(f"  Total Time: {profile['total_time']:.3f}s")
    print(f"  Transitions: {profile['transitions']}")
    print(f"  Avg Transition Time: {profile['avg_transition_time']:.3f}s")
    
    if 'state_times' in profile:
        print("\n  State Timings:")
        for state, timing in profile['state_times'].items():
            print(f"    {state}: {timing['avg']:.3f}s (count: {timing['count']})")


def main():
    """Run the advanced FSM demonstrations."""
    print("Advanced FSM Example - Simplified")
    print("=" * 70)
    print("This example demonstrates the actual AdvancedFSM API features")
    
    # Run synchronous demonstration
    demonstrate_advanced_features()
    
    # Run async demonstration
    print("\n" + "=" * 70)
    print("Running Async Execution Demo...")
    asyncio.run(demonstrate_async_execution())
    
    print("\n" + "=" * 70)
    print("Example Complete!")


if __name__ == "__main__":
    main()