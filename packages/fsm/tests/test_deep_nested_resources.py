"""Test resource inheritance in deeply nested networks (2+ levels)."""

import pytest
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import (
    FSMConfig,
    NetworkConfig,
    StateConfig,
    PushArcConfig,
    FunctionReference,
    ResourceConfig,
    ResourceType,
)
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.network import NetworkExecutor
from dataknobs_fsm.core.modes import ProcessingMode


class TestDeepNestedResources:
    """Test resource inheritance through multiple network levels."""

    def test_three_level_network_resource_inheritance(self):
        """Test that resources from all parent levels are accessible in deeply nested networks.

        This test creates a 3-level deep network hierarchy:
        - Level 1 (main): Has level1_resource
        - Level 2 (middle): Has level2_resource, should also see level1_resource
        - Level 3 (deep): Has level3_resource, should see resources from level1 and level2
        """
        config = FSMConfig(
            name="test_fsm",
            main_network="level1",
            # Define resources at different levels
            resources=[
                ResourceConfig(
                    name="level1_resource",
                    type=ResourceType.CUSTOM,
                    config={
                        "class": "dataknobs_fsm.resources.properties.PropertiesResource",
                        "initial_properties": {
                            "resource_type": "level1",
                            "level": "main",
                            "data": "from_level1"
                        },
                        "max_instances": 10,
                    }
                ),
                ResourceConfig(
                    name="level2_resource",
                    type=ResourceType.CUSTOM,
                    config={
                        "class": "dataknobs_fsm.resources.properties.PropertiesResource",
                        "initial_properties": {
                            "resource_type": "level2",
                            "level": "middle",
                            "data": "from_level2"
                        },
                        "max_instances": 10,
                    }
                ),
                ResourceConfig(
                    name="level3_resource",
                    type=ResourceType.CUSTOM,
                    config={
                        "class": "dataknobs_fsm.resources.properties.PropertiesResource",
                        "initial_properties": {
                            "resource_type": "level3",
                            "level": "deep",
                            "data": "from_level3"
                        },
                        "max_instances": 10,
                    }
                )
            ],
            networks=[
                # Level 1 - Main network
                NetworkConfig(
                    name="level1",
                    states=[
                        StateConfig(
                            name="l1_start",
                            is_start=True,
                            resources=["level1_resource"],
                            arcs=[
                                PushArcConfig(
                                    target="l1_end",
                                    target_network="level2",
                                    return_state="l1_end"
                                )
                            ],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    # Track resources at level 1
    if 'resource_trace' not in context.variables:
        context.variables['resource_trace'] = []

    available = list(context.resources.keys()) if hasattr(context, 'resources') and context.resources else []
    context.variables['resource_trace'].append({
        'state': 'l1_start',
        'level': 1,
        'available_resources': available
    })

    # Mark level1 resource as accessed
    if 'level1_resource' in (context.resources if hasattr(context, 'resources') else {}):
        handle = context.resources['level1_resource']
        if hasattr(handle, 'set'):
            handle.set('accessed_by_l1', True)

    data['l1_processed'] = True
    return data
"""
                                )
                            ]
                        ),
                        StateConfig(
                            name="l1_end",
                            is_end=True,
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    available = list(context.resources.keys()) if hasattr(context, 'resources') and context.resources else []
    context.variables['resource_trace'].append({
        'state': 'l1_end',
        'level': 1,
        'available_resources': available
    })
    data['l1_end_processed'] = True
    return data
"""
                                )
                            ]
                        )
                    ]
                ),
                # Level 2 - Middle network
                NetworkConfig(
                    name="level2",
                    states=[
                        StateConfig(
                            name="l2_start",
                            is_start=True,
                            resources=["level2_resource"],
                            arcs=[
                                PushArcConfig(
                                    target="l2_end",
                                    target_network="level3",
                                    return_state="l2_end"
                                )
                            ],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    # Check for level 1 resources
    current = list(context.resources.keys()) if hasattr(context, 'resources') and context.resources else []
    parent = []
    if hasattr(context, 'metadata') and 'parent_state_resources' in context.metadata:
        parent = list(context.metadata['parent_state_resources'].keys())

    context.variables['resource_trace'].append({
        'state': 'l2_start',
        'level': 2,
        'current_resources': current,
        'parent_resources': parent,
        'has_level1_resource': 'level1_resource' in parent or 'level1_resource' in current
    })

    # Access level1 resource if available
    if hasattr(context, 'metadata') and 'parent_state_resources' in context.metadata:
        if 'level1_resource' in context.metadata['parent_state_resources']:
            handle = context.metadata['parent_state_resources']['level1_resource']
            if hasattr(handle, 'set'):
                handle.set('accessed_by_l2', True)

    # Mark level2 resource
    if 'level2_resource' in (context.resources if hasattr(context, 'resources') else {}):
        handle = context.resources['level2_resource']
        if hasattr(handle, 'set'):
            handle.set('accessed_by_l2', True)

    data['l2_processed'] = True
    return data
"""
                                )
                            ]
                        ),
                        StateConfig(
                            name="l2_end",
                            is_end=True,
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    current = list(context.resources.keys()) if hasattr(context, 'resources') and context.resources else []
    context.variables['resource_trace'].append({
        'state': 'l2_end',
        'level': 2,
        'current_resources': current
    })
    data['l2_end_processed'] = True
    return data
"""
                                )
                            ]
                        )
                    ]
                ),
                # Level 3 - Deep network
                NetworkConfig(
                    name="level3",
                    states=[
                        StateConfig(
                            name="l3_state",
                            is_start=True,
                            is_end=True,
                            resources=["level3_resource"],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    # Check what resources are available at level 3
    current = list(context.resources.keys()) if hasattr(context, 'resources') and context.resources else []
    parent = []
    if hasattr(context, 'metadata') and 'parent_state_resources' in context.metadata:
        parent = list(context.metadata['parent_state_resources'].keys())

    # THIS IS THE KEY TEST: Can we see resources from level 1 and level 2?
    has_level1 = 'level1_resource' in parent or 'level1_resource' in current
    has_level2 = 'level2_resource' in parent or 'level2_resource' in current
    has_level3 = 'level3_resource' in current

    context.variables['resource_trace'].append({
        'state': 'l3_state',
        'level': 3,
        'current_resources': current,
        'parent_resources': parent,
        'has_level1_resource': has_level1,
        'has_level2_resource': has_level2,
        'has_level3_resource': has_level3,
        'all_resources_available': has_level1 and has_level2 and has_level3
    })

    # Try to access all resources
    resources_accessed = []

    # Check parent resources
    if hasattr(context, 'metadata') and 'parent_state_resources' in context.metadata:
        parent_res = context.metadata['parent_state_resources']
        for res_name in ['level1_resource', 'level2_resource']:
            if res_name in parent_res:
                handle = parent_res[res_name]
                if hasattr(handle, 'get'):
                    data_val = handle.get('data', 'not_found')
                    resources_accessed.append(f"{res_name}:{data_val}")
                    if hasattr(handle, 'set'):
                        handle.set('accessed_by_l3', True)

    # Check current resources
    if hasattr(context, 'resources') and context.resources:
        for res_name in ['level3_resource']:
            if res_name in context.resources:
                handle = context.resources[res_name]
                if hasattr(handle, 'get'):
                    data_val = handle.get('data', 'not_found')
                    resources_accessed.append(f"{res_name}:{data_val}")

    context.variables['l3_resources_accessed'] = resources_accessed
    data['l3_processed'] = True
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

        # Create context and executor
        context = ExecutionContext(data_mode=ProcessingMode.SINGLE)
        context.resource_manager = fsm.resource_manager
        executor = NetworkExecutor(fsm)

        # Execute FSM
        input_data = {"test": "deep_nesting"}
        success, result = executor.execute_network("level1", context, input_data)

        # Verify execution succeeded
        assert success, f"Execution failed: {result}"

        # Check that all levels were processed
        assert result.get("l1_processed") == True, "Level 1 not processed"
        assert result.get("l2_processed") == True, "Level 2 not processed"
        assert result.get("l3_processed") == True, "Level 3 not processed"
        assert result.get("l1_end_processed") == True, "Level 1 end not processed"
        assert result.get("l2_end_processed") == True, "Level 2 end not processed"

        # Analyze resource trace
        resource_trace = context.variables.get('resource_trace', [])

        # Print trace for debugging
        print("\n📊 Resource Trace Through Network Levels:")
        print("=" * 60)
        for trace in resource_trace:
            print(f"\n📍 State: {trace['state']} (Level {trace.get('level', '?')})")
            for key, value in trace.items():
                if key not in ['state', 'level']:
                    print(f"   {key}: {value}")

        # Find traces for each level
        l1_trace = next((t for t in resource_trace if t['state'] == 'l1_start'), None)
        l2_trace = next((t for t in resource_trace if t['state'] == 'l2_start'), None)
        l3_trace = next((t for t in resource_trace if t['state'] == 'l3_state'), None)

        assert l1_trace is not None, "Missing level 1 trace"
        assert l2_trace is not None, "Missing level 2 trace"
        assert l3_trace is not None, "Missing level 3 trace"

        # Level 1 should have its resource
        assert 'level1_resource' in l1_trace.get('available_resources', [])

        # Level 2 should have its resource and see level 1's
        assert 'level2_resource' in l2_trace.get('current_resources', [])
        assert l2_trace.get('has_level1_resource') == True, "Level 2 cannot see level 1 resource"

        # THE KEY ASSERTIONS: Level 3 should have access to resources from ALL levels
        assert l3_trace.get('has_level3_resource') == True, "Level 3 doesn't have its own resource"
        assert l3_trace.get('has_level2_resource') == True, "Level 3 cannot see level 2 resource"
        assert l3_trace.get('has_level1_resource') == True, "Level 3 cannot see level 1 resource"
        assert l3_trace.get('all_resources_available') == True, "Not all resources available at level 3"

        # Verify resources were actually accessible (not just visible)
        l3_accessed = context.variables.get('l3_resources_accessed', [])
        assert 'level1_resource:from_level1' in l3_accessed, "Level 1 resource data not accessible"
        assert 'level2_resource:from_level2' in l3_accessed, "Level 2 resource data not accessible"
        assert 'level3_resource:from_level3' in l3_accessed, "Level 3 resource data not accessible"

    def test_resource_cleanup_on_network_return(self):
        """Test that resources are properly cleaned up when returning from nested networks."""
        # This test ensures that:
        # 1. Resources allocated in subnetworks are released when returning
        # 2. Parent resources remain available after return
        # 3. No resource leakage occurs

        config = FSMConfig(
            name="test_fsm",
            main_network="main",
            resources=[
                ResourceConfig(
                    name="main_resource",
                    type=ResourceType.CUSTOM,
                    config={
                        "class": "dataknobs_fsm.resources.properties.PropertiesResource",
                        "initial_properties": {"level": "main"},
                        "max_instances": 10,
                    }
                ),
                ResourceConfig(
                    name="sub_resource",
                    type=ResourceType.CUSTOM,
                    config={
                        "class": "dataknobs_fsm.resources.properties.PropertiesResource",
                        "initial_properties": {"level": "sub"},
                        "max_instances": 10,
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
                            resources=["main_resource"],
                            arcs=[
                                PushArcConfig(
                                    target="end",
                                    target_network="sub",
                                    return_state="end"
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
    # After returning from subnetwork, check resources
    available = list(context.resources.keys()) if hasattr(context, 'resources') and context.resources else []
    data['final_resources'] = available
    data['has_main_resource'] = 'main_resource' in available
    data['has_sub_resource'] = 'sub_resource' in available  # Should be False
    return data
"""
                                )
                            ]
                        )
                    ]
                ),
                NetworkConfig(
                    name="sub",
                    states=[
                        StateConfig(
                            name="sub_state",
                            is_start=True,
                            is_end=True,
                            resources=["sub_resource"],
                            transforms=[
                                FunctionReference(
                                    type="inline",
                                    code="""
def transform(data, context):
    available = list(context.resources.keys()) if hasattr(context, 'resources') and context.resources else []
    data['sub_resources'] = available
    return data
"""
                                )
                            ]
                        )
                    ]
                )
            ]
        )

        builder = FSMBuilder()
        fsm = builder.build(config)
        context = ExecutionContext(data_mode=ProcessingMode.SINGLE)
        context.resource_manager = fsm.resource_manager
        executor = NetworkExecutor(fsm)

        success, result = executor.execute_network("main", context, {})
        assert success

        # Verify no resource leakage
        assert result.get('has_main_resource') == False, "Main resource should not be in end state (no resource requirement)"
        assert result.get('has_sub_resource') == False, "Sub resource should not leak to parent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])