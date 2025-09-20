"""Tests for properties resource module."""

from datetime import datetime
import pytest

from dataknobs_fsm.resources.properties import PropertiesHandle, PropertiesResource
from dataknobs_fsm.resources.base import ResourceStatus, ResourceHealth
from dataknobs_fsm.functions.base import ResourceError


class TestPropertiesHandle:
    """Test suite for PropertiesHandle class."""

    def test_handle_initialization(self):
        """Test PropertiesHandle initialization."""
        initial_props = {"key1": "value1", "key2": 42}
        handle = PropertiesHandle(
            resource_name="test_resource",
            owner_id="test_owner",
            instance_id="instance_1",
            properties=initial_props
        )

        assert handle.resource_name == "test_resource"
        assert handle.owner_id == "test_owner"
        assert handle.instance_id == "instance_1"
        assert handle.properties == initial_props
        assert handle.properties is not initial_props  # Should be a copy
        assert handle.access_count == 0
        assert len(handle.modifications) == 0
        assert isinstance(handle.created_at, datetime)
        assert isinstance(handle.accessed_at, datetime)

    def test_handle_get(self):
        """Test getting property values."""
        handle = PropertiesHandle(
            resource_name="test",
            owner_id="owner",
            instance_id="inst1",
            properties={"key1": "value1", "key2": 42}
        )

        # Test existing key
        assert handle.get("key1") == "value1"
        assert handle.get("key2") == 42
        assert handle.access_count == 2

        # Test non-existing key with default
        assert handle.get("key3") is None
        assert handle.get("key3", "default") == "default"
        assert handle.access_count == 4

    def test_handle_set(self):
        """Test setting property values."""
        handle = PropertiesHandle(
            resource_name="test",
            owner_id="owner",
            instance_id="inst1",
            properties={"key1": "value1"}
        )

        # Set new value
        handle.set("key2", "value2")
        assert handle.properties["key2"] == "value2"
        assert len(handle.modifications) == 1

        mod = handle.modifications[0]
        assert mod["key"] == "key2"
        assert mod["old_value"] is None
        assert mod["new_value"] == "value2"

        # Update existing value
        handle.set("key1", "updated_value")
        assert handle.properties["key1"] == "updated_value"
        assert len(handle.modifications) == 2

        mod = handle.modifications[1]
        assert mod["key"] == "key1"
        assert mod["old_value"] == "value1"
        assert mod["new_value"] == "updated_value"

    def test_handle_update(self):
        """Test updating multiple properties."""
        handle = PropertiesHandle(
            resource_name="test",
            owner_id="owner",
            instance_id="inst1",
            properties={"key1": "value1"}
        )

        updates = {"key2": "value2", "key3": 3, "key1": "updated"}
        handle.update(updates)

        assert handle.properties["key1"] == "updated"
        assert handle.properties["key2"] == "value2"
        assert handle.properties["key3"] == 3
        assert len(handle.modifications) == 3

    def test_handle_delete(self):
        """Test deleting properties."""
        handle = PropertiesHandle(
            resource_name="test",
            owner_id="owner",
            instance_id="inst1",
            properties={"key1": "value1", "key2": "value2"}
        )

        # Delete existing key
        result = handle.delete("key1")
        assert result == "value1"
        assert "key1" not in handle.properties
        assert len(handle.modifications) == 1

        mod = handle.modifications[0]
        assert mod["key"] == "key1"
        assert mod["old_value"] == "value1"
        assert mod["new_value"] is None
        assert mod["operation"] == "delete"

        # Delete non-existing key
        result = handle.delete("key_nonexistent")
        assert result is None
        assert len(handle.modifications) == 1  # No new modification

    def test_handle_clear(self):
        """Test clearing all properties."""
        handle = PropertiesHandle(
            resource_name="test",
            owner_id="owner",
            instance_id="inst1",
            properties={"key1": "value1", "key2": "value2"}
        )

        handle.clear()
        assert len(handle.properties) == 0
        assert len(handle.modifications) == 1
        assert handle.modifications[0]["operation"] == "clear"

    def test_handle_to_dict(self):
        """Test converting handle to dictionary."""
        handle = PropertiesHandle(
            resource_name="test",
            owner_id="owner",
            instance_id="inst1",
            properties={"key1": "value1"}
        )

        handle.set("key2", "value2")
        handle.get("key1")  # Increase access count

        result = handle.to_dict()

        assert result["resource_name"] == "test"
        assert result["owner_id"] == "owner"
        assert result["instance_id"] == "inst1"
        assert result["properties"] == {"key1": "value1", "key2": "value2"}
        assert result["access_count"] == 1
        assert result["modification_count"] == 1
        assert "created_at" in result
        assert "accessed_at" in result


class TestPropertiesResource:
    """Test suite for PropertiesResource class."""

    def test_resource_initialization(self):
        """Test PropertiesResource initialization."""
        initial_props = {"default_key": "default_value"}
        resource = PropertiesResource(
            name="test_resource",
            initial_properties=initial_props,
            max_instances=5,
            track_history=True
        )

        assert resource.name == "test_resource"
        assert resource.initial_properties == initial_props
        assert resource.max_instances == 5
        assert resource.track_history is True
        assert resource.status == ResourceStatus.IDLE
        assert len(resource._instances) == 0

    def test_resource_acquire_basic(self):
        """Test basic resource acquisition."""
        resource = PropertiesResource(
            name="test_resource",
            initial_properties={"key1": "value1"}
        )

        handle = resource.acquire(owner_id="test_owner")

        assert isinstance(handle, PropertiesHandle)
        assert handle.owner_id == "test_owner"
        assert handle.properties["key1"] == "value1"
        assert "_metadata" in handle.properties
        assert handle.properties["_metadata"]["resource_name"] == "test_resource"
        assert resource.status == ResourceStatus.ACTIVE
        assert len(resource._instances) == 1

    def test_resource_acquire_with_properties(self):
        """Test resource acquisition with additional properties."""
        resource = PropertiesResource(
            name="test_resource",
            initial_properties={"key1": "initial"}
        )

        handle = resource.acquire(
            owner_id="test_owner",
            properties={"key2": "added", "key1": "overridden"}
        )

        assert handle.properties["key1"] == "overridden"
        assert handle.properties["key2"] == "added"

    def test_resource_acquire_max_instances(self):
        """Test resource acquisition with max instances limit."""
        resource = PropertiesResource(
            name="test_resource",
            max_instances=2
        )

        # Acquire max instances
        handle1 = resource.acquire(owner_id="owner1")
        handle2 = resource.acquire(owner_id="owner2")

        assert len(resource._instances) == 2

        # Try to acquire beyond limit
        with pytest.raises(ResourceError) as exc_info:
            resource.acquire(owner_id="owner3")

        assert "Maximum instances" in str(exc_info.value)
        assert exc_info.value.resource_name == "test_resource"

    def test_resource_release_by_handle(self):
        """Test releasing resource by handle."""
        resource = PropertiesResource(name="test_resource")

        handle = resource.acquire(owner_id="owner1")
        assert len(resource._instances) == 1
        assert resource.status == ResourceStatus.ACTIVE

        resource.release(handle)
        assert len(resource._instances) == 0
        assert resource.status == ResourceStatus.IDLE

    def test_resource_release_by_instance_id(self):
        """Test releasing resource by instance ID."""
        resource = PropertiesResource(name="test_resource")

        handle = resource.acquire(owner_id="owner1")
        instance_id = handle.instance_id

        resource.release(instance_id)
        assert len(resource._instances) == 0
        assert resource.status == ResourceStatus.IDLE

    def test_resource_release_invalid(self):
        """Test releasing with invalid resource."""
        resource = PropertiesResource(name="test_resource")

        # Release non-existent instance ID
        resource.release("non_existent_id")  # Should not raise, just log warning

        # Release invalid type
        resource.release(123)  # Should not raise, just log warning

    def test_resource_get_instance(self):
        """Test getting specific instance."""
        resource = PropertiesResource(name="test_resource")

        handle1 = resource.acquire(owner_id="owner1")
        handle2 = resource.acquire(owner_id="owner2")

        retrieved = resource.get_instance(handle1.instance_id)
        assert retrieved is handle1

        none_result = resource.get_instance("non_existent")
        assert none_result is None

    def test_resource_get_all_instances(self):
        """Test getting all instances."""
        resource = PropertiesResource(name="test_resource")

        handle1 = resource.acquire(owner_id="owner1")
        handle2 = resource.acquire(owner_id="owner2")

        all_instances = resource.get_all_instances()
        assert len(all_instances) == 2
        assert handle1.instance_id in all_instances
        assert handle2.instance_id in all_instances

    def test_resource_health_check(self):
        """Test resource health check."""
        resource = PropertiesResource(name="test_resource", max_instances=10)

        # Healthy (0 instances)
        assert resource.health_check() == ResourceHealth.HEALTHY

        # Still healthy (< 80%)
        for i in range(7):
            resource.acquire(owner_id=f"owner{i}")
        assert resource.health_check() == ResourceHealth.HEALTHY

        # Degraded (>= 80%)
        resource.acquire(owner_id="owner8")
        assert resource.health_check() == ResourceHealth.DEGRADED

        # Unhealthy (100%)
        resource.acquire(owner_id="owner9")
        resource.acquire(owner_id="owner10")
        assert resource.health_check() == ResourceHealth.UNHEALTHY

    def test_resource_history_tracking(self):
        """Test resource history tracking."""
        resource = PropertiesResource(
            name="test_resource",
            track_history=True
        )

        handle = resource.acquire(owner_id="owner1", properties={"key": "value"})
        handle.set("new_key", "new_value")
        handle.get("key")  # Increase access count

        assert len(resource._acquisition_history) == 1
        acq_history = resource._acquisition_history[0]
        assert acq_history["owner_id"] == "owner1"
        assert acq_history["instance_id"] == handle.instance_id

        resource.release(handle)
        assert len(resource._release_history) == 1
        rel_history = resource._release_history[0]
        assert rel_history["owner_id"] == "owner1"
        assert rel_history["access_count"] == 1
        assert rel_history["modification_count"] == 1

    def test_resource_no_history_tracking(self):
        """Test resource without history tracking."""
        resource = PropertiesResource(
            name="test_resource",
            track_history=False
        )

        handle = resource.acquire(owner_id="owner1")
        assert len(resource._acquisition_history) == 0

        resource.release(handle)
        assert len(resource._release_history) == 0

    def test_resource_get_stats(self):
        """Test getting resource statistics."""
        resource = PropertiesResource(
            name="test_resource",
            max_instances=10
        )

        handle1 = resource.acquire(owner_id="owner1")
        handle2 = resource.acquire(owner_id="owner2")
        handle1.set("key", "value")

        stats = resource.get_stats()

        assert stats["name"] == "test_resource"
        assert stats["status"] == ResourceStatus.ACTIVE.value
        assert stats["active_instances"] == 2
        assert stats["max_instances"] == 10
        assert len(stats["instance_ids"]) == 2
        assert "instances" in stats
        assert "metrics" in stats

    def test_resource_reset(self):
        """Test resetting resource."""
        resource = PropertiesResource(
            name="test_resource",
            track_history=True
        )

        # Create some state
        handle1 = resource.acquire(owner_id="owner1")
        handle2 = resource.acquire(owner_id="owner2")
        resource.release(handle1)

        assert len(resource._instances) == 1
        assert len(resource._acquisition_history) == 2
        assert len(resource._release_history) == 1

        # Reset
        resource.reset()

        assert len(resource._instances) == 0
        assert len(resource._acquisition_history) == 0
        assert len(resource._release_history) == 0
        assert resource._instance_counter == 0

    def test_resource_repr(self):
        """Test resource string representation."""
        resource = PropertiesResource(
            name="test_resource",
            max_instances=5
        )

        resource.acquire(owner_id="owner1")
        resource.acquire(owner_id="owner2")

        repr_str = repr(resource)
        assert "PropertiesResource" in repr_str
        assert "name='test_resource'" in repr_str
        assert "instances=2/5" in repr_str
        assert f"status={ResourceStatus.ACTIVE.value}" in repr_str

    def test_resource_metrics_tracking(self):
        """Test that metrics are properly tracked."""
        resource = PropertiesResource(name="test_resource")

        # Initially no metrics
        metrics = resource.get_metrics()
        assert metrics.total_acquisitions == 0

        # Acquire and release
        handle = resource.acquire(owner_id="owner1")
        metrics = resource.get_metrics()
        assert metrics.total_acquisitions == 1
        assert metrics.active_connections == 1

        resource.release(handle)
        metrics = resource.get_metrics()
        assert metrics.active_connections == 0
