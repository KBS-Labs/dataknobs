"""Test resource inheritance in multi-state subnetworks using PropertiesCustomResource."""

import logging
from pathlib import Path
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import (
    FSMConfig,
    NetworkConfig,
    StateConfig,
    ArcConfig,
    PushArcConfig,
    FunctionReference,
    ResourceConfig,
    ResourceType,
)
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.network import NetworkExecutor
from dataknobs_fsm.core.modes import ProcessingMode

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_multistate_subnetwork_resource_inheritance():
    """Test that parent state resources are available to all states in a subnetwork."""

    config = FSMConfig(
        name="test_fsm",
        main_network="main",
        # Define PropertiesCustomResource resources at FSM level
        resources=[
            ResourceConfig(
                name="parent_resource",
                type=ResourceType.CUSTOM,
                config={
                    "class": "dataknobs_fsm.resources.properties.PropertiesResource",
                    "initial_properties": {
                        "resource_type": "parent",
                        "level": "main",
                        "purpose": "testing parent resource inheritance"
                    },
                    "max_instances": 10,
                    "track_history": True
                }
            ),
            ResourceConfig(
                name="sub_resource",
                type=ResourceType.CUSTOM,
                config={
                    "class": "dataknobs_fsm.resources.properties.PropertiesResource",
                    "initial_properties": {
                        "resource_type": "sub",
                        "level": "subnetwork",
                        "purpose": "testing subnetwork-specific resources"
                    },
                    "max_instances": 10,
                    "track_history": True
                }
            )
        ],
        networks=[
            NetworkConfig(
                name="main",
                states=[
                    StateConfig(
                        name="start",
                        is_start=True,
                        resources=["parent_resource"],  # This state needs parent_resource
                        arcs=[
                            PushArcConfig(
                                target="end",  # Target state in same network
                                target_network="sub_network",  # Push to subnetwork
                                return_state="end"
                            )
                        ],
                        transforms=[
                            FunctionReference(
                                type="inline",
                                code="""
def transform(data, context):
    # Initialize tracking
    if 'resource_trace' not in context.variables:
        context.variables['resource_trace'] = []

    # Check what resources are allocated to this state
    available_resources = []
    resource_details = {}

    # Transform functions receive FunctionContext, not ExecutionContext
    # Resources are in context.resources
    if hasattr(context, 'resources') and context.resources:
        available_resources = list(context.resources.keys())
        # Get details from the PropertiesHandle
        for name, handle in context.resources.items():
            if hasattr(handle, 'properties'):
                resource_details[name] = {
                    'instance_id': handle.instance_id,
                    'resource_type': handle.get('resource_type', 'unknown'),
                    'level': handle.get('level', 'unknown')
                }

    context.variables['resource_trace'].append({
        'state': 'start',
        'available_resources': available_resources,
        'resource_details': resource_details,
        'has_parent_resource': 'parent_resource' in available_resources
    })

    # Mark the resource as accessed from this state
    if 'parent_resource' in (context.resources if hasattr(context, 'resources') else {}):
        handle = context.resources['parent_resource']
        if hasattr(handle, 'set'):
            handle.set('accessed_by_start', True)
            handle.set('start_state_data', data.get('test', 'no_data'))

    data['parent_processed'] = True
    return data
"""
                            )
                        ]
                    ),
                    StateConfig(
                        name="end",
                        is_end=True,
                        transforms=[
                            FunctionReference(
                                type="inline",
                                code="""
def transform(data, context):
    # Check resources after returning from subnetwork
    available_resources = []
    if hasattr(context, 'resources') and context.resources:
        available_resources = list(context.resources.keys())

    context.variables['resource_trace'].append({
        'state': 'end',
        'available_resources': available_resources,
        'still_has_parent_resource': 'parent_resource' in available_resources,
        'has_sub_resource': 'sub_resource' in available_resources  # Should be False
    })

    # Check if parent resource was modified by subnetwork
    if 'parent_resource' in (context.resources if hasattr(context, 'resources') else {}):
        handle = context.resources['parent_resource']
        if hasattr(handle, 'get'):
            was_accessed_by_sub = handle.get('accessed_by_subnetwork', False)
            context.variables['parent_resource_modified_by_sub'] = was_accessed_by_sub

    data['end_processed'] = True
    return data
"""
                            )
                        ]
                    )
                ]
            ),
            NetworkConfig(
                name="sub_network",
                states=[
                    StateConfig(
                        name="sub_start",
                        is_start=True,
                        arcs=[
                            ArcConfig(
                                target="sub_middle"
                            )
                        ],
                        transforms=[
                            FunctionReference(
                                type="inline",
                                code="""
def transform(data, context):
    # Check for parent resources in first subnetwork state
    current_resources = []
    parent_resources = []

    # Transform functions receive FunctionContext with resources
    if hasattr(context, 'resources') and context.resources:
        current_resources = list(context.resources.keys())

    # Parent resources are in metadata
    if hasattr(context, 'metadata') and 'parent_state_resources' in context.metadata:
        parent_resources = list(context.metadata['parent_state_resources'].keys())
        # Access parent resource to verify it works
        if 'parent_resource' in context.metadata['parent_state_resources']:
            handle = context.metadata['parent_state_resources']['parent_resource']
            if hasattr(handle, 'set'):
                handle.set('accessed_by_subnetwork', True)
                handle.set('sub_start_visited', True)

    context.variables['resource_trace'].append({
        'state': 'sub_start',
        'current_resources': current_resources,
        'parent_resources': parent_resources,
        'has_parent_resources': len(parent_resources) > 0,
        'parent_resource_accessible': 'parent_resource' in parent_resources
    })

    data['sub_start_processed'] = True
    return data
"""
                            )
                        ]
                    ),
                    StateConfig(
                        name="sub_middle",
                        resources=["sub_resource"],  # This state needs its own resource
                        arcs=[
                            ArcConfig(
                                target="sub_end"
                            )
                        ],
                        transforms=[
                            FunctionReference(
                                type="inline",
                                code="""
def transform(data, context):
    current_resources = []
    parent_resources = []

    # Check current state resources (should include BOTH parent and sub resources)
    if hasattr(context, 'resources') and context.resources:
        current_resources = list(context.resources.keys())

        # Mark sub_resource as used
        if 'sub_resource' in context.resources:
            handle = context.resources['sub_resource']
            if hasattr(handle, 'set'):
                handle.set('used_in_sub_middle', True)
                handle.set('processing_data', data)

    # Parent resources are in metadata
    if hasattr(context, 'metadata') and 'parent_state_resources' in context.metadata:
        parent_resources = list(context.metadata['parent_state_resources'].keys())
        # Access parent resource
        if 'parent_resource' in context.metadata['parent_state_resources']:
            handle = context.metadata['parent_state_resources']['parent_resource']
            if hasattr(handle, 'set'):
                handle.set('sub_middle_visited', True)

    # In _allocate_state_resources, parent resources should be merged with current
    # So we expect parent_resource to be in current_resources due to merging
    resources_properly_merged = (
        'parent_resource' in current_resources and
        'sub_resource' in current_resources
    )

    context.variables['resource_trace'].append({
        'state': 'sub_middle',
        'current_resources': current_resources,
        'parent_resources': parent_resources,
        'has_sub_resource': 'sub_resource' in current_resources,
        'has_parent_resource_in_current': 'parent_resource' in current_resources,
        'has_parent_resource_in_parent': 'parent_resource' in parent_resources,
        'resources_properly_merged': resources_properly_merged
    })

    data['sub_middle_processed'] = True
    return data
"""
                            )
                        ]
                    ),
                    StateConfig(
                        name="sub_end",
                        is_end=True,
                        transforms=[
                            FunctionReference(
                                type="inline",
                                code="""
def transform(data, context):
    current_resources = []
    parent_resources = []

    if hasattr(context, 'resources') and context.resources:
        current_resources = list(context.resources.keys())

    # Parent resources are in metadata
    if hasattr(context, 'metadata') and 'parent_state_resources' in context.metadata:
        parent_resources = list(context.metadata['parent_state_resources'].keys())
        # Final access to parent resource
        if 'parent_resource' in context.metadata['parent_state_resources']:
            handle = context.metadata['parent_state_resources']['parent_resource']
            if hasattr(handle, 'set'):
                handle.set('sub_end_visited', True)
                handle.set('subnetwork_completed', True)

    context.variables['resource_trace'].append({
        'state': 'sub_end',
        'current_resources': current_resources,
        'parent_resources': parent_resources,
        'parent_still_accessible': 'parent_resource' in parent_resources
    })

    data['sub_end_processed'] = True
    return data
"""
                            )
                        ]
                    )
                ]
            )
        ]
    )

    # Build FSM
    builder = FSMBuilder()
    fsm = builder.build(config)

    # Resources should be registered in the FSM's resource manager

    # Create context and executor
    context = ExecutionContext(data_mode=ProcessingMode.SINGLE)
    # Critical: Ensure context has resource manager from FSM!
    context.resource_manager = fsm.resource_manager
    executor = NetworkExecutor(fsm)

    # Ensure FSM has a resource manager for the test

    # Execute FSM
    input_data = {"test": "data"}
    success, result = executor.execute_network("main", context, input_data)

    # Verify execution succeeded
    assert success, f"Execution failed: {result}"

    # Context variables should contain the resource trace

    # Check that all expected states were processed
    assert result.get("parent_processed") == True, "Parent state not processed"
    assert result.get("sub_start_processed") == True, "Sub-start state not processed"
    assert result.get("sub_middle_processed") == True, "Sub-middle state not processed"
    assert result.get("sub_end_processed") == True, "Sub-end state not processed"
    assert result.get("end_processed") == True, "End state not processed"

    # Analyze resource trace
    resource_trace = context.variables.get('resource_trace', [])

    print("\n📊 Resource Trace Through States:")
    print("=" * 60)
    for trace_entry in resource_trace:
        state = trace_entry['state']
        print(f"\n📍 State: {state}")
        for key, value in trace_entry.items():
            if key != 'state':
                print(f"   {key}: {value}")

    # Find entries for each state
    start_trace = next((t for t in resource_trace if t['state'] == 'start'), None)
    sub_start_trace = next((t for t in resource_trace if t['state'] == 'sub_start'), None)
    sub_middle_trace = next((t for t in resource_trace if t['state'] == 'sub_middle'), None)
    sub_end_trace = next((t for t in resource_trace if t['state'] == 'sub_end'), None)
    end_trace = next((t for t in resource_trace if t['state'] == 'end'), None)

    print("\n🧪 Test Assertions:")
    print("=" * 60)

    # Verify parent allocated resources
    assert start_trace is not None, "Missing start state trace"
    assert start_trace.get('has_parent_resource', False) == True, "Start state should have parent_resource"
    print("✅ Start state has parent_resource")

    # Verify all subnetwork states have access to parent resources
    assert sub_start_trace is not None, "Missing sub_start state trace"
    assert sub_start_trace.get('parent_resource_accessible', False) == True, "sub_start should have access to parent_resource"
    print("✅ Sub-start has access to parent_resource")

    assert sub_middle_trace is not None, "Missing sub_middle state trace"
    assert sub_middle_trace.get('has_sub_resource', False) == True, "sub_middle should have sub_resource"
    print("✅ Sub-middle has its own sub_resource")

    # The key test: parent resources should be merged into current_state_resources
    assert sub_middle_trace.get('resources_properly_merged', False) == True, "sub_middle should have both resources merged in current_state_resources"
    print("✅ Sub-middle has both parent and sub resources properly merged")

    assert sub_end_trace is not None, "Missing sub_end state trace"
    assert sub_end_trace.get('parent_still_accessible', False) == True, "sub_end should still have access to parent_resource"
    print("✅ Sub-end still has access to parent_resource")

    # Verify parent state doesn't have sub resources after return
    assert end_trace is not None, "Missing end state trace"
    assert end_trace.get('has_sub_resource', False) == False, "sub_resource should not leak to parent"
    print("✅ No sub_resource leakage to parent state")

    # Check if parent resource was modified by subnetwork
    if context.variables.get('parent_resource_modified_by_sub'):
        print("✅ Parent resource was successfully accessed and modified by subnetwork")

    # Get resource manager stats if available
    if hasattr(fsm, 'resource_manager'):
        print("\n📈 Resource Manager Statistics:")
        print("=" * 60)
        for resource_name in ['parent_resource', 'sub_resource']:
            provider = fsm.resource_manager.get_provider(resource_name)
            if provider and hasattr(provider, 'get_stats'):
                stats = provider.get_stats()
                print(f"\n{resource_name}:")
                print(f"  Total acquisitions: {stats.get('total_acquisitions', 0)}")
                print(f"  Total releases: {stats.get('total_releases', 0)}")
                print(f"  Active instances: {stats.get('active_instances', 0)}")

    print("\n" + "=" * 60)
    print("✅ All multi-state subnetwork resource inheritance tests passed!")
    print("=" * 60)
    print("\nKey findings:")
    print("  • Parent state resources are properly allocated")
    print("  • Parent resources are accessible to ALL subnetwork states")
    print("  • State-specific resources are properly allocated and merged")
    print("  • No resource leakage from subnetwork back to parent")
    print("  • Resources can be shared and modified across network boundaries")


if __name__ == "__main__":
    test_multistate_subnetwork_resource_inheritance()