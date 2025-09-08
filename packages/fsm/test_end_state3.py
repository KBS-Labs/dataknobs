"""Debug FSM structure."""
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

print("FSM wrapper type:", type(fsm._fsm))
print("FSM wrapper main_network:", fsm._fsm.main_network)
print("FSM wrapper main_network type:", type(fsm._fsm.main_network))

# Check core FSM
if hasattr(fsm._fsm, 'core_fsm'):
    print("\nCore FSM main_network:", fsm._fsm.core_fsm.main_network)
    print("Core FSM main_network type:", type(fsm._fsm.core_fsm.main_network))
    
    # The engine uses the core FSM
    print("\nEngine's FSM:", fsm._engine.fsm)
    print("Engine's FSM main_network:", fsm._engine.fsm.main_network)
    print("Engine's FSM main_network type:", type(fsm._engine.fsm.main_network))
