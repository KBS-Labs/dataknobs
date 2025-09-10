#!/usr/bin/env python3
"""Debug config transformation to verify state transforms are properly assigned."""

from dataknobs_fsm.config.loader import ConfigLoader

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

print("Original config:")
print(f"Multiply state: {config['networks'][0]['states'][1]}")

loader = ConfigLoader()
fsm_config = loader.load_from_dict(config)

print("\nAfter transformation:")
for state in fsm_config.networks[0].states:
    if state.name == 'multiply':
        print(f"Multiply state transforms: {state.transforms}")
        print(f"Multiply state arcs: {state.arcs}")
    elif state.name == 'input':
        print(f"Input state arcs: {state.arcs}")