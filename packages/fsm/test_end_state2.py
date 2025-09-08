"""Debug end state recognition with more detail."""
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.execution.engine import ExecutionEngine

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

print("FSM main_network:", fsm._fsm.main_network)
print("FSM networks:", list(fsm._fsm.networks.keys()))

# Check engine's _is_final_state
engine = fsm._engine
for state_name in ['input', 'multiply', 'output']:
    is_final = engine._is_final_state(state_name)
    print(f"  engine._is_final_state('{state_name}') = {is_final}")

# Process data
input_data = {'value': 5}
result = fsm.process(input_data)

print(f"\nResult: {result}")
