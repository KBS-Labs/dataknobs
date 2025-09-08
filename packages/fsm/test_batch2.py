"""Test batch execution result structure."""
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.execution.batch import BatchExecutor

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
results = batch_executor.execute_batch(items=items)

print(f"Results type: {type(results)}")
if results:
    result = results[0]
    print(f"\nFirst result:")
    print(f"  index: {result.index}")
    print(f"  success: {result.success}")
    print(f"  result: {result.result}")
    print(f"  error: {result.error}")
    print(f"  metadata: {result.metadata}")
