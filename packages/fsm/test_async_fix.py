"""Debug async execution."""
from dataknobs_fsm.api.simple import SimpleFSM

config = {
    'name': 'test_fsm',
    'main_network': 'main',
    'networks': [{
        'name': 'main',
        'states': [
            {'name': 'input', 'is_start': True},
            {'name': 'multiply', 'functions': {'transform': 'lambda state: {"result": state.data.get("value", 1) * 2}'}},
            {'name': 'output', 'is_end': True}
        ],
        'arcs': [
            {'from': 'input', 'to': 'multiply', 'name': 'process'},
            {'from': 'multiply', 'to': 'output', 'name': 'finish'}
        ]
    }]
}

fsm = SimpleFSM(config)
result = fsm.process({'value': 5})
print(f"Result: {result}")
print(f"Success: {result['success']}")
if not result['success']:
    print(f"Error: {result['error']}")
