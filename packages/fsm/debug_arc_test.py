#!/usr/bin/env python3
"""Debug arc execution to see what happens when arc has no transform."""

from dataknobs_fsm.config.loader import ConfigLoader
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.core.arc import ArcExecution
from dataknobs_fsm.execution.context import ExecutionContext

# Create the same FSM as debug test
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

# Build FSM
loader = ConfigLoader()
fsm_config = loader.load_from_dict(config)
builder = FSMBuilder()
fsm = builder.build(fsm_config)

# Get the arc from input to multiply
input_state = fsm.get_state('input')
print(f"Input state: {input_state}")
print(f"Input state arcs: {input_state.outgoing_arcs}")

if input_state.outgoing_arcs:
    arc_def = input_state.outgoing_arcs[0]
    print(f"Arc definition: {arc_def}")
    print(f"Arc transform: {arc_def.transform}")
    
    # Create arc execution
    func_reg = {}
    arc_exec = ArcExecution(
        arc_def,
        source_state="input",
        function_registry=func_reg
    )
    
    # Create execution context
    context = ExecutionContext()
    context.data = {'value': 5}
    
    print(f"Initial data: {context.data}")
    
    # Try to execute the arc
    try:
        result = arc_exec.execute(context, context.data)
        print(f"Arc execution result: {result}")
        print(f"Result is None: {result is None}")
    except Exception as e:
        print(f"Arc execution error: {e}")
        import traceback
        traceback.print_exc()