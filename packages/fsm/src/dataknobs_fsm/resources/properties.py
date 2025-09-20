"""Properties custom resource for testing and demonstration.

This module provides a simple dictionary-based resource that can be used
for testing FSM resource management and as an example of custom resources.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List
from dataknobs_fsm.resources.base import BaseResourceProvider, ResourceStatus, ResourceHealth
from dataknobs_fsm.functions.base import ResourceError

logger = logging.getLogger(__name__)


class PropertiesHandle:
    """Handle for a properties resource instance."""

    def __init__(self, resource_name: str, owner_id: str, instance_id: str, properties: Dict[str, Any]):
        """Initialize properties handle.

        Args:
            resource_name: Name of the resource.
            owner_id: ID of the owner.
            instance_id: Unique instance identifier.
            properties: Initial properties dictionary.
        """
        self.resource_name = resource_name
        self.owner_id = owner_id
        self.instance_id = instance_id
        self.properties = properties.copy()  # Create a copy to avoid external mutations
        self.created_at = datetime.now()
        self.accessed_at = datetime.now()
        self.access_count = 0
        self.modifications = []

    def get(self, key: str, default: Any = None) -> Any:
        """Get a property value.

        Args:
            key: Property key.
            default: Default value if key not found.

        Returns:
            Property value or default.
        """
        self.accessed_at = datetime.now()
        self.access_count += 1
        return self.properties.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a property value.

        Args:
            key: Property key.
            value: Property value.
        """
        old_value = self.properties.get(key)
        self.properties[key] = value
        self.accessed_at = datetime.now()
        self.modifications.append({
            'timestamp': datetime.now(),
            'key': key,
            'old_value': old_value,
            'new_value': value
        })

    def update(self, properties: Dict[str, Any]) -> None:
        """Update multiple properties.

        Args:
            properties: Dictionary of properties to update.
        """
        for key, value in properties.items():
            self.set(key, value)

    def delete(self, key: str) -> Any:
        """Delete a property.

        Args:
            key: Property key to delete.

        Returns:
            The deleted value or None.
        """
        if key in self.properties:
            value = self.properties.pop(key)
            self.modifications.append({
                'timestamp': datetime.now(),
                'key': key,
                'old_value': value,
                'new_value': None,
                'operation': 'delete'
            })
            return value
        return None

    def clear(self) -> None:
        """Clear all properties."""
        self.properties.clear()
        self.modifications.append({
            'timestamp': datetime.now(),
            'operation': 'clear'
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert handle to dictionary representation.

        Returns:
            Dictionary with handle information.
        """
        return {
            'resource_name': self.resource_name,
            'owner_id': self.owner_id,
            'instance_id': self.instance_id,
            'properties': self.properties.copy(),
            'created_at': self.created_at.isoformat(),
            'accessed_at': self.accessed_at.isoformat(),
            'access_count': self.access_count,
            'modification_count': len(self.modifications)
        }


class PropertiesResource(BaseResourceProvider):
    """Resource provider that manages property dictionaries.

    This resource is useful for:
    - Testing resource management in FSMs
    - Storing configuration or state data
    - Demonstrating custom resource implementation
    - Tracking resource lifecycle and access patterns
    """

    def __init__(
        self,
        name: str,
        initial_properties: Dict[str, Any] | None = None,
        max_instances: int = 10,
        track_history: bool = True,
        **config
    ):
        """Initialize properties resource.

        Args:
            name: Resource name.
            initial_properties: Initial properties for all instances.
            max_instances: Maximum number of concurrent instances.
            track_history: Whether to track acquisition/release history.
            **config: Additional configuration.
        """
        super().__init__(name, config)

        # Configuration
        self.initial_properties = initial_properties or {}
        self.max_instances = max_instances
        self.track_history = track_history

        # Instance tracking
        self._instances: Dict[str, PropertiesHandle] = {}
        self._instance_counter = 0

        # History tracking
        self._acquisition_history: List[Dict[str, Any]] = []
        self._release_history: List[Dict[str, Any]] = []

        # State
        self.status = ResourceStatus.IDLE
        logger.info(f"Initialized PropertiesResource '{name}' with max_instances={max_instances}")

    def acquire(
        self,
        owner_id: str | None = None,
        properties: Dict[str, Any] | None = None,
        **kwargs
    ) -> PropertiesHandle:
        """Acquire a properties resource instance.

        Args:
            owner_id: ID of the owner (e.g., state name).
            properties: Additional properties to merge with initial properties.
            **kwargs: Additional parameters (ignored).

        Returns:
            PropertiesHandle instance.

        Raises:
            ResourceError: If maximum instances exceeded.
        """
        # Check instance limit
        if len(self._instances) >= self.max_instances:
            raise ResourceError(
                f"Maximum instances ({self.max_instances}) exceeded",
                resource_name=self.name,
                operation="acquire"
            )

        # Generate instance ID
        self._instance_counter += 1
        instance_id = f"{self.name}_instance_{self._instance_counter}"
        owner_id = owner_id or "unknown"

        # Merge properties
        merged_properties = self.initial_properties.copy()
        if properties:
            merged_properties.update(properties)

        # Add metadata
        merged_properties['_metadata'] = {
            'resource_name': self.name,
            'instance_id': instance_id,
            'owner_id': owner_id,
            'acquired_at': datetime.now().isoformat()
        }

        # Create handle
        handle = PropertiesHandle(
            resource_name=self.name,
            owner_id=owner_id,
            instance_id=instance_id,
            properties=merged_properties
        )

        # Track instance
        self._instances[instance_id] = handle
        self._resources.append(handle)

        # Track history
        if self.track_history:
            self._acquisition_history.append({
                'timestamp': datetime.now(),
                'instance_id': instance_id,
                'owner_id': owner_id,
                'properties': merged_properties.copy()
            })

        # Update status
        self.status = ResourceStatus.ACTIVE
        self.metrics.record_acquisition()

        logger.debug(f"Acquired properties resource: {instance_id} for owner: {owner_id}")
        return handle

    def release(self, resource: Any) -> None:
        """Release a properties resource instance.

        Args:
            resource: PropertiesHandle or instance_id to release.
        """
        # Handle different input types
        if isinstance(resource, PropertiesHandle):
            instance_id = resource.instance_id
            handle = resource
        elif isinstance(resource, str):
            instance_id = resource
            handle = self._instances.get(instance_id)
        else:
            logger.warning(f"Unknown resource type for release: {type(resource)}")
            return

        # Remove from tracking
        if instance_id in self._instances:
            del self._instances[instance_id]

        if handle and handle in self._resources:
            self._resources.remove(handle)

            # Track history
            if self.track_history:
                self._release_history.append({
                    'timestamp': datetime.now(),
                    'instance_id': instance_id,
                    'owner_id': handle.owner_id,
                    'final_properties': handle.properties.copy(),
                    'access_count': handle.access_count,
                    'modification_count': len(handle.modifications)
                })

            self.metrics.record_release(
                (datetime.now() - handle.created_at).total_seconds()
            )

            logger.debug(f"Released properties resource: {instance_id}")

        # Update status
        if not self._instances:
            self.status = ResourceStatus.IDLE

    def get_instance(self, instance_id: str) -> PropertiesHandle | None:
        """Get a specific instance by ID.

        Args:
            instance_id: Instance identifier.

        Returns:
            PropertiesHandle or None if not found.
        """
        return self._instances.get(instance_id)

    def get_all_instances(self) -> Dict[str, PropertiesHandle]:
        """Get all active instances.

        Returns:
            Dictionary of instance_id -> PropertiesHandle.
        """
        return self._instances.copy()

    def health_check(self) -> ResourceHealth:
        """Check resource health.

        Returns:
            Resource health status.
        """
        if len(self._instances) >= self.max_instances:
            return ResourceHealth.UNHEALTHY
        elif len(self._instances) >= self.max_instances * 0.8:
            return ResourceHealth.DEGRADED
        else:
            return ResourceHealth.HEALTHY

    def get_stats(self) -> Dict[str, Any]:
        """Get resource statistics.

        Returns:
            Dictionary with resource statistics.
        """
        stats = {
            'name': self.name,
            'status': self.status.value,
            'health': self.health_check().value,
            'active_instances': len(self._instances),
            'max_instances': self.max_instances,
            'total_acquisitions': len(self._acquisition_history),
            'total_releases': len(self._release_history),
            'instance_ids': list(self._instances.keys())
        }

        # Add instance details
        if self._instances:
            stats['instances'] = {
                instance_id: handle.to_dict()
                for instance_id, handle in self._instances.items()
            }

        # Add metrics
        metrics = self.get_metrics()
        stats['metrics'] = {
            'total_acquisitions': metrics.total_acquisitions,
            'active_connections': metrics.active_connections,
            'failed_acquisitions': metrics.failed_acquisitions,
            'average_hold_time': metrics.average_hold_time,
            'average_acquisition_time': metrics.average_acquisition_time
        }

        return stats

    def reset(self) -> None:
        """Reset the resource, releasing all instances."""
        # Release all instances
        for instance_id in list(self._instances.keys()):
            self.release(instance_id)

        # Clear history if needed
        self._acquisition_history.clear()
        self._release_history.clear()
        self._instance_counter = 0

        logger.info(f"Reset PropertiesResource '{self.name}'")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PropertiesResource(name='{self.name}', "
            f"instances={len(self._instances)}/{self.max_instances}, "
            f"status={self.status.value})"
        )
