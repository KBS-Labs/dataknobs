"""Tests for the unified function manager."""

import asyncio
import pytest
from typing import Any, Dict, Optional

from dataknobs_fsm.functions.manager import (
    FunctionManager,
    FunctionWrapper,
    FunctionSource,
    InterfaceWrapper,
    get_function_manager,
    register_function,
    resolve_function
)
from dataknobs_fsm.functions.base import (
    ITransformFunction,
    IValidationFunction,
    IStateTestFunction,
    ExecutionResult
)


# Test functions
def sync_transform(data: Dict[str, Any], context=None) -> Dict[str, Any]:
    """A synchronous transform function."""
    data['transformed'] = True
    return data


async def async_transform(data: Dict[str, Any], context=None) -> Dict[str, Any]:
    """An asynchronous transform function."""
    await asyncio.sleep(0)  # Simulate async work
    data['async_transformed'] = True
    return data


def sync_test(data: Dict[str, Any], context=None) -> bool:
    """A synchronous test function."""
    return data.get('value', 0) > 0


async def async_test(data: Dict[str, Any], context=None) -> tuple:
    """An asynchronous test function."""
    await asyncio.sleep(0)
    value = data.get('value', 0)
    return (value > 0, f"Value is {value}")


class CallableClass:
    """A callable class for testing."""

    def __call__(self, data: Dict[str, Any], context=None) -> Dict[str, Any]:
        data['called'] = True
        return data


class AsyncCallableClass:
    """An async callable class for testing."""

    async def __call__(self, data: Dict[str, Any], context=None) -> Dict[str, Any]:
        await asyncio.sleep(0)
        data['async_called'] = True
        return data


class TestFunctionWrapper:
    """Test FunctionWrapper class."""

    def test_sync_function_wrapper(self):
        """Test wrapping a sync function."""
        wrapper = FunctionWrapper(sync_transform, "sync_transform")

        assert wrapper.name == "sync_transform"
        assert not wrapper.is_async
        assert wrapper.source == FunctionSource.REGISTERED

        # Test sync execution
        result = wrapper.execute_sync({'value': 1}, None)
        assert result['transformed'] is True

    @pytest.mark.asyncio
    async def test_async_function_wrapper(self):
        """Test wrapping an async function."""
        wrapper = FunctionWrapper(async_transform, "async_transform")

        assert wrapper.name == "async_transform"
        assert wrapper.is_async
        assert wrapper.source == FunctionSource.REGISTERED

        # Test async execution
        result = await wrapper.execute_async({'value': 1}, None)
        assert result['async_transformed'] is True

    def test_sync_callable_wrapper(self):
        """Test wrapping a callable class."""
        callable_obj = CallableClass()
        wrapper = FunctionWrapper(callable_obj, "callable_class")

        assert not wrapper.is_async
        result = wrapper.execute_sync({'value': 1}, None)
        assert result['called'] is True

    @pytest.mark.asyncio
    async def test_async_callable_wrapper(self):
        """Test wrapping an async callable class."""
        callable_obj = AsyncCallableClass()
        wrapper = FunctionWrapper(callable_obj, "async_callable")

        assert wrapper.is_async
        result = await wrapper.execute_async({'value': 1}, None)
        assert result['async_called'] is True

    def test_async_function_sync_execution_error(self):
        """Test that async functions raise error on sync execution."""
        wrapper = FunctionWrapper(async_transform, "async_transform")

        with pytest.raises(RuntimeError, match="Cannot execute async function"):
            wrapper.execute_sync({'value': 1}, None)

    @pytest.mark.asyncio
    async def test_sync_function_async_execution(self):
        """Test that sync functions can be executed async."""
        wrapper = FunctionWrapper(sync_transform, "sync_transform")

        result = await wrapper.execute_async({'value': 1}, None)
        assert result['transformed'] is True

    @pytest.mark.asyncio
    async def test_wrapper_call_preserves_async(self):
        """Test that calling wrapper preserves async nature."""
        async_wrapper = FunctionWrapper(async_transform, "async")
        sync_wrapper = FunctionWrapper(sync_transform, "sync")

        # Async wrapper returns coroutine
        result_coro = async_wrapper({'value': 1}, None)
        assert asyncio.iscoroutine(result_coro)
        result = await result_coro
        assert result['async_transformed'] is True

        # Sync wrapper returns direct result
        result = sync_wrapper({'value': 1}, None)
        assert result['transformed'] is True


class TestInterfaceWrapper:
    """Test InterfaceWrapper class."""

    def test_transform_interface_sync(self):
        """Test adapting sync function to transform interface."""
        wrapper = FunctionWrapper(sync_transform, "sync_transform")
        interface_wrapper = InterfaceWrapper(wrapper, ITransformFunction)

        assert hasattr(interface_wrapper, 'transform')
        assert hasattr(interface_wrapper, 'get_transform_description')

        result = interface_wrapper.transform({'value': 1}, None)
        assert isinstance(result, ExecutionResult)
        assert result.success
        assert result.data['transformed'] is True

    @pytest.mark.asyncio
    async def test_transform_interface_async(self):
        """Test adapting async function to transform interface."""
        wrapper = FunctionWrapper(async_transform, "async_transform")
        interface_wrapper = InterfaceWrapper(wrapper, ITransformFunction)

        assert interface_wrapper.is_async
        result = await interface_wrapper.transform({'value': 1}, None)
        assert isinstance(result, ExecutionResult)
        assert result.success
        assert result.data['async_transformed'] is True

    def test_state_test_interface_sync(self):
        """Test adapting sync function to state test interface."""
        wrapper = FunctionWrapper(sync_test, "sync_test")
        interface_wrapper = InterfaceWrapper(wrapper, IStateTestFunction)

        assert hasattr(interface_wrapper, 'test')
        assert hasattr(interface_wrapper, 'get_test_description')

        result = interface_wrapper.test({'value': 5}, None)
        assert result == (True, None)

        result = interface_wrapper.test({'value': -5}, None)
        assert result == (False, None)

    @pytest.mark.asyncio
    async def test_state_test_interface_async(self):
        """Test adapting async function to state test interface."""
        wrapper = FunctionWrapper(async_test, "async_test")
        interface_wrapper = InterfaceWrapper(wrapper, IStateTestFunction)

        assert interface_wrapper.is_async
        result = await interface_wrapper.test({'value': 5}, None)
        assert result == (True, "Value is 5")

        result = await interface_wrapper.test({'value': -5}, None)
        assert result == (False, "Value is -5")


class TestFunctionManager:
    """Test FunctionManager class."""

    def test_register_sync_function(self):
        """Test registering a sync function."""
        manager = FunctionManager()
        wrapper = manager.register_function("sync", sync_transform)

        assert wrapper.name == "sync"
        assert not wrapper.is_async
        assert manager.has_function("sync")

    def test_register_async_function(self):
        """Test registering an async function."""
        manager = FunctionManager()
        wrapper = manager.register_function("async", async_transform)

        assert wrapper.name == "async"
        assert wrapper.is_async
        assert manager.has_function("async")

    def test_register_multiple_functions(self):
        """Test registering multiple functions."""
        manager = FunctionManager()
        functions = {
            'sync': sync_transform,
            'async': async_transform,
            'test': sync_test
        }

        wrappers = manager.register_functions(functions)
        assert len(wrappers) == 3
        assert not wrappers['sync'].is_async
        assert wrappers['async'].is_async

    def test_register_builtin_function(self):
        """Test registering a builtin function."""
        manager = FunctionManager()
        wrapper = manager.register_function(
            "builtin",
            sync_transform,
            source=FunctionSource.BUILTIN
        )

        assert wrapper.source == FunctionSource.BUILTIN
        assert manager.has_function("builtin")

    def test_resolve_callable(self):
        """Test resolving a direct callable."""
        manager = FunctionManager()
        wrapper = manager.resolve_function(sync_transform)

        assert wrapper is not None
        assert not wrapper.is_async

    def test_resolve_registered_string(self):
        """Test resolving a registered function by name."""
        manager = FunctionManager()
        manager.register_function("test", sync_transform)

        wrapper = manager.resolve_function("test")
        assert wrapper is not None
        assert wrapper.name == "test"

    def test_resolve_inline_string(self):
        """Test resolving inline code string."""
        manager = FunctionManager()
        code = "data['inline'] = True; return data"

        wrapper = manager.resolve_function(code)
        assert wrapper is not None
        assert wrapper.source == FunctionSource.INLINE

        result = wrapper.execute_sync({'value': 1}, None)
        assert result['inline'] is True

    def test_resolve_dict_registered(self):
        """Test resolving a dictionary reference to registered function."""
        manager = FunctionManager()
        manager.register_function("test", sync_transform)

        reference = {'type': 'registered', 'name': 'test'}
        wrapper = manager.resolve_function(reference)

        assert wrapper is not None
        assert wrapper.name == "test"

    def test_resolve_dict_inline(self):
        """Test resolving a dictionary reference to inline code."""
        manager = FunctionManager()
        reference = {
            'type': 'inline',
            'code': "data['from_dict'] = True; return data"
        }

        wrapper = manager.resolve_function(reference)
        assert wrapper is not None
        assert wrapper.source == FunctionSource.INLINE

        result = wrapper.execute_sync({'value': 1}, None)
        assert result['from_dict'] is True

    def test_resolve_with_interface(self):
        """Test resolving with interface adaptation."""
        manager = FunctionManager()
        manager.register_function("transform", sync_transform)

        wrapper = manager.resolve_function("transform", ITransformFunction)
        assert wrapper is not None
        assert hasattr(wrapper, 'transform')
        assert hasattr(wrapper, 'get_transform_description')

    def test_get_function(self):
        """Test getting a function by name."""
        manager = FunctionManager()
        manager.register_function("test", sync_transform)

        wrapper = manager.get_function("test")
        assert wrapper is not None
        assert wrapper.name == "test"

        wrapper = manager.get_function("nonexistent")
        assert wrapper is None

    def test_list_functions(self):
        """Test listing all functions."""
        manager = FunctionManager()
        manager.register_function("sync", sync_transform)
        manager.register_function("async", async_transform)
        manager.register_function(
            "builtin",
            sync_test,
            source=FunctionSource.BUILTIN
        )

        functions = manager.list_functions()
        assert len(functions) == 3
        assert functions['sync']['async'] is False
        assert functions['async']['async'] is True
        assert functions['builtin']['type'] == 'builtin'

    def test_clear_functions(self):
        """Test clearing functions."""
        manager = FunctionManager()
        manager.register_function("test", sync_transform)
        manager.register_function(
            "builtin",
            sync_test,
            source=FunctionSource.BUILTIN
        )

        manager.clear()
        assert not manager.has_function("test")
        assert manager.has_function("builtin")  # Builtins preserved

        manager.clear_all()
        assert not manager.has_function("builtin")

    def test_inline_cache(self):
        """Test that inline functions are cached."""
        manager = FunctionManager()
        code = "data['cached'] = True; return data"

        wrapper1 = manager.resolve_function(code)
        wrapper2 = manager.resolve_function(code)

        # Should be the same wrapper instance (cached)
        assert wrapper1 is wrapper2


class TestGlobalFunctionManager:
    """Test global function manager functions."""

    def test_global_manager_singleton(self):
        """Test that global manager is a singleton."""
        manager1 = get_function_manager()
        manager2 = get_function_manager()
        assert manager1 is manager2

    def test_global_register_function(self):
        """Test global register function."""
        wrapper = register_function("global_test", sync_transform)
        assert wrapper.name == "global_test"

        manager = get_function_manager()
        assert manager.has_function("global_test")

    def test_global_resolve_function(self):
        """Test global resolve function."""
        register_function("resolve_test", async_transform)

        wrapper = resolve_function("resolve_test")
        assert wrapper is not None
        assert wrapper.is_async


@pytest.mark.asyncio
async def test_integration_async_workflow():
    """Test complete async workflow with manager."""
    manager = FunctionManager()

    # Register various function types
    manager.register_function("sync_t", sync_transform)
    manager.register_function("async_t", async_transform)
    manager.register_function("sync_test", sync_test)
    manager.register_function("async_test", async_test)

    # Test data
    data = {'value': 10}

    # Resolve and execute sync transform
    sync_wrapper = manager.resolve_function("sync_t", ITransformFunction)
    result = sync_wrapper.transform(data.copy(), None)
    assert result.success
    assert result.data['transformed'] is True

    # Resolve and execute async transform
    async_wrapper = manager.resolve_function("async_t", ITransformFunction)
    result = await async_wrapper.transform(data.copy(), None)
    assert result.success
    assert result.data['async_transformed'] is True

    # Resolve and execute sync test
    test_wrapper = manager.resolve_function("sync_test", IStateTestFunction)
    test_result = test_wrapper.test(data, None)
    assert test_result == (True, None)

    # Resolve and execute async test
    async_test_wrapper = manager.resolve_function("async_test", IStateTestFunction)
    test_result = await async_test_wrapper.test(data, None)
    assert test_result == (True, "Value is 10")


def test_integration_mixed_sources():
    """Test integration with mixed function sources."""
    manager = FunctionManager()

    # Register different types
    manager.register_function("registered", sync_transform)
    manager.register_function(
        "builtin",
        sync_test,
        source=FunctionSource.BUILTIN
    )

    # Test inline code
    inline_code = """
def process(data, context=None):
    data['processed'] = True
    return data
"""
    inline_wrapper = manager.resolve_function(inline_code)
    result = inline_wrapper.execute_sync({'value': 1}, None)
    assert result['processed'] is True

    # Test registered
    reg_wrapper = manager.get_function("registered")
    result = reg_wrapper.execute_sync({'value': 1}, None)
    assert result['transformed'] is True

    # Test builtin
    builtin_wrapper = manager.get_function("builtin")
    result = builtin_wrapper.execute_sync({'value': 5}, None)
    assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
