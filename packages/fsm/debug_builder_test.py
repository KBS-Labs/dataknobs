#!/usr/bin/env python3
"""Debug FSMBuilder to verify state transforms are properly processed."""

from dataknobs_fsm.config.loader import ConfigLoader
from dataknobs_fsm.config.builder import FSMBuilder

# Simple test configuration with state functions
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

print("Testing ConfigLoader + FSMBuilder pipeline...")

# Load config
loader = ConfigLoader()
fsm_config = loader.load_from_dict(config)

# Build FSM
builder = FSMBuilder()
fsm = builder.build(fsm_config)

print(f"FSM created: {fsm}")
print(f"Main network: {fsm.main_network}")

# Get the multiply state
multiply_state = fsm.get_state('multiply')
print(f"Multiply state: {multiply_state}")
if multiply_state:
    print(f"  has transform_functions: {hasattr(multiply_state, 'transform_functions')}")
    if hasattr(multiply_state, 'transform_functions'):
        print(f"  transform_functions: {multiply_state.transform_functions}")
        
        # Test the transform function
        if multiply_state.transform_functions:
            transform_func = multiply_state.transform_functions[0]
            print(f"  transform_func: {transform_func}")
            print(f"  callable: {callable(transform_func)}")
            
            # Test it
            from types import SimpleNamespace
            test_state = SimpleNamespace(data={'value': 5})
            try:
                result = transform_func(test_state)
                print(f"  transform result: {result}")
            except Exception as e:
                print(f"  transform error: {e}")
                
                # Try calling with just the data instead
                try:
                    result = transform_func({'value': 5})
                    print(f"  transform result (data only): {result}")
                except Exception as e2:
                    print(f"  transform error (data only): {e2}")