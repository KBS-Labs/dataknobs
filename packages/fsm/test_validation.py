"""Test validation."""
from dataknobs_fsm.api.simple import SimpleFSM

config = {
    'name': 'test_simple_fsm',
    'main_network': 'main',
    'networks': [{
        'name': 'main',
        'states': [
            {'name': 'start', 'is_start': True},
            {'name': 'process', 'functions': {'transform': 'lambda state: {"output": state.data["input"].upper(), "processed": True}'}},
            {'name': 'end', 'is_end': True}
        ],
        'arcs': [
            {'from': 'start', 'to': 'process', 'name': 'begin_processing'},
            {'from': 'process', 'to': 'end', 'name': 'complete'}
        ]
    }]
}

fsm = SimpleFSM(config)

# Check start state
start_state = fsm._fsm.get_start_state()
print(f"Start state: {start_state.name}")
print(f"Start state schema: {start_state.schema}")

# Valid data
valid_result = fsm.validate({'input': 'test_string'})
print(f"Valid result: {valid_result}")

# Invalid data (missing required field)
invalid_result = fsm.validate({})
print(f"Invalid result: {invalid_result}")
