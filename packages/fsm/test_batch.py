"""Test batch execution."""
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.execution.batch import BatchExecutor
import asyncio

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
batch_executor = BatchExecutor(
    fsm=fsm._fsm,
    parallelism=4,
    batch_size=10
)

# Test batch execution directly
items = [{'value': 1}, {'value': 2}, {'value': 3}]
results = asyncio.run(batch_executor.execute_batch(items=items))

print(f"Results type: {type(results)}")
if results:
    print(f"First result type: {type(results[0])}")
    print(f"First result attributes: {dir(results[0])}")
    print(f"First result: {results[0]}")
