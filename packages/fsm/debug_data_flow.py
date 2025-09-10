#!/usr/bin/env python3
"""Debug data flow to see what happens to the input data."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataknobs_fsm.execution.engine import ExecutionEngine

# Store the original method
original_execute_state_transforms = ExecutionEngine._execute_state_transforms

def debug_execute_state_transforms(self, context, state_name):
    """Debug wrapper to see what data the state transforms receive."""
    print(f"\n=== DEBUGGING _execute_state_transforms ===")
    print(f"State: {state_name}")
    print(f"Context data: {context.data}")
    print(f"Context data type: {type(context.data)}")
    
    # Get the state definition
    state_def = self.fsm.get_state(state_name)
    if not state_def:
        print("No state definition found")
        return
        
    print(f"State def: {state_def}")
    print(f"Has transform_functions: {hasattr(state_def, 'transform_functions')}")
    
    # Execute any transform functions defined on the state
    if hasattr(state_def, 'transform_functions') and state_def.transform_functions:
        print(f"Number of transform functions: {len(state_def.transform_functions)}")
        for i, transform_func in enumerate(state_def.transform_functions):
            print(f"Transform function {i}: {transform_func}")
            try:
                # Create function context
                from dataknobs_fsm.functions.base import FunctionContext
                func_context = FunctionContext(
                    state_name=state_name,
                    function_name=getattr(transform_func, '__name__', 'transform'),
                    metadata={'state': state_name},
                    resources={}
                )
                
                # Execute the transform
                # Create a mock state object for transforms that expect state.data
                from types import SimpleNamespace
                state_obj = SimpleNamespace(data=context.data)
                
                print(f"Creating state_obj with data: {state_obj.data}")
                print(f"state_obj.data type: {type(state_obj.data)}")
                
                # Try calling with state object first (for inline lambdas)
                try:
                    print("Trying to call transform with state object...")
                    result = transform_func(state_obj)
                    print(f"Transform result: {result}")
                except (TypeError, AttributeError) as e:
                    print(f"Failed with state object: {e}")
                    # Fall back to calling with data and context
                    print("Trying with data and context...")
                    result = transform_func(context.data, func_context)
                    print(f"Transform result (fallback): {result}")
                
                if result is not None:
                    print(f"Updating context.data from {context.data} to {result}")
                    context.data = result
                else:
                    print("Transform returned None, not updating context")
                    
            except Exception as e:
                print(f"Transform failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("No transform functions to execute")

# Monkey patch
ExecutionEngine._execute_state_transforms = debug_execute_state_transforms

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