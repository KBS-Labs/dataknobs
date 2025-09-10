#!/usr/bin/env python3
"""Debug to catch the actual exception being thrown."""

# Monkey patch the execution engine to capture exceptions
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataknobs_fsm.execution.engine import ExecutionEngine

# Store the original method
original_execute_transition = ExecutionEngine._execute_transition

def debug_execute_transition(self, context, arc):
    """Debug wrapper to catch exceptions."""
    print(f"\n=== DEBUGGING _execute_transition ===")
    print(f"Arc: {arc}")
    print(f"Context state: {context.current_state}")
    print(f"Context data: {context.data}")
    
    try:
        result = original_execute_transition(self, context, arc)
        print(f"Transition result: {result}")
        return result
    except Exception as e:
        print(f"EXCEPTION in _execute_transition: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

# Monkey patch
ExecutionEngine._execute_transition = debug_execute_transition

# Now run the test
from dataknobs_fsm.api.simple import SimpleFSM

config = {
    'name': 'debug_fsm',
    'main_network': 'main',
    'networks': [{
        'name': 'main',
        'states': [
            {
                'name': 'input',
                'is_start': True
            },
            {
                'name': 'multiply',
                'functions': {
                    'transform': 'lambda state: {"result": state.data.get("value", 1) * 2}'
                }
            },
            {
                'name': 'output',
                'is_end': True
            }
        ],
        'arcs': [
            {'from': 'input', 'to': 'multiply', 'name': 'process'},
            {'from': 'multiply', 'to': 'output', 'name': 'done'}
        ]
    }]
}

print("Creating FSM...")
fsm = SimpleFSM(config)

print("Calling fsm.process()...")
result = fsm.process({'value': 5})

print(f"\nFinal result: {result}")