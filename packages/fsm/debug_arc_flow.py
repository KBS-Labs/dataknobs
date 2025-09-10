#!/usr/bin/env python3
"""Debug what happens during arc traversal."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataknobs_fsm.execution.engine import ExecutionEngine

# Store the original method
original_execute_transition = ExecutionEngine._execute_transition

def debug_execute_transition(self, context, arc):
    """Debug wrapper to trace the complete transition."""
    print(f"\n=== DEBUGGING COMPLETE TRANSITION ===")
    print(f"Arc: {arc}")
    print(f"From state: {context.current_state}")
    print(f"To state: {arc.target_state}")
    print(f"Context data BEFORE transition: {context.data}")
    
    # Call the original method
    result = original_execute_transition(self, context, arc)
    
    print(f"Context data AFTER transition: {context.data}")
    print(f"Transition result: {result}")
    print(f"Current state after transition: {context.current_state}")
    
    return result

# Monkey patch
ExecutionEngine._execute_transition = debug_execute_transition

# Also debug arc execution
from dataknobs_fsm.core.arc import ArcExecution

original_arc_execute = ArcExecution.execute

def debug_arc_execute(self, context, data=None, stream_enabled=False):
    """Debug wrapper for arc execution."""
    print(f"\n    === ARC EXECUTION ===")
    print(f"    Arc transform: {self.arc_def.transform}")
    print(f"    Input data: {data}")
    
    result = original_arc_execute(self, context, data, stream_enabled)
    
    print(f"    Arc execution result: {result}")
    return result

# Monkey patch arc execution too
ArcExecution.execute = debug_arc_execute

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

print("Testing execution with value=5...")
result = fsm.process({'value': 5})

print(f"\nFinal result: {result}")