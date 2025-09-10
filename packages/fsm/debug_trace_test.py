#!/usr/bin/env python3
"""Debug execution with detailed tracing."""

from dataknobs_fsm.api.simple import SimpleFSM

# Simple test configuration
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

print(f"Final result: {result}")

# Let's look more closely at the execution engine
print(f"\nEngine type: {type(fsm._engine)}")
print(f"Engine error count: {fsm._engine._error_count}")
print(f"Engine transition count: {fsm._engine._transition_count}")

# Let's try to manually execute the FSM
print("\n=== Manual execution ===")
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.core.modes import ProcessingMode
from dataknobs_fsm.core.context_factory import ContextFactory

context = ContextFactory.create_context(
    fsm=fsm._fsm,
    data={'value': 5},
    initial_state=None,
    data_mode=ProcessingMode.SINGLE,
    resource_manager=fsm._resource_manager
)

print(f"Initial context state: {context.current_state}")
print(f"Initial context data: {context.data}")

# Try one step of execution
success, step_result = fsm._engine.execute(context, None, 1)  # Only 1 transition
print(f"One step result: success={success}, result={step_result}")
print(f"Context state after 1 step: {context.current_state}")
print(f"Context data after 1 step: {context.data}")