"""Debug end state recognition."""
from dataknobs_fsm.api.simple import SimpleFSM

# Create FSM with END state
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

# Debug: Check states
print("States in FSM:")
for network in fsm._fsm.networks.values():
    for state in network.states.values():
        print(f"  {state.name}: is_end_state={state.is_end_state()}, type={state.type}")

# Process data
input_data = {'value': 5}
result = fsm.process(input_data)

print(f"\nResult: {result}")
