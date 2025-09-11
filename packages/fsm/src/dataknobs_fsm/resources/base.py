"""Base interfaces and classes for resource management."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Protocol, runtime_checkable
from contextlib import contextmanager


class ResourceStatus(Enum):
    """Status of a resource."""
    
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    CLOSED = "closed"
    MAINTENANCE = "maintenance"


class ResourceHealth(Enum):
    """Health status of a resource."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ResourceMetrics:
    """Metrics for resource usage."""
    
    total_acquisitions: int = 0
    active_connections: int = 0
    failed_acquisitions: int = 0
    average_hold_time: float = 0.0
    last_acquisition_time: datetime | None = None
    last_release_time: datetime | None = None
    health_check_failures: int = 0
    last_health_check: datetime | None = None
    average_acquisition_time: float = 0.0
    total_timeout_events: int = 0
    last_timeout_time: datetime | None = None
    
    def record_acquisition(self, acquisition_time: float | None = None) -> None:
        """Record a resource acquisition.
        
        Args:
            acquisition_time: Time taken to acquire the resource in seconds.
        """
        self.total_acquisitions += 1
        self.active_connections += 1
        self.last_acquisition_time = datetime.now()
        
        # Update average acquisition time if provided
        if acquisition_time is not None:
            if self.average_acquisition_time == 0:
                self.average_acquisition_time = acquisition_time
            else:
                # Rolling average
                self.average_acquisition_time = (self.average_acquisition_time * 0.9) + (acquisition_time * 0.1)
    
    def record_release(self, hold_time: float) -> None:
        """Record a resource release.
        
        Args:
            hold_time: How long the resource was held in seconds.
        """
        self.active_connections = max(0, self.active_connections - 1)
        self.last_release_time = datetime.now()
        
        # Update average hold time
        if self.average_hold_time == 0:
            self.average_hold_time = hold_time
        else:
            # Rolling average
            self.average_hold_time = (self.average_hold_time * 0.9) + (hold_time * 0.1)
    
    def record_failure(self) -> None:
        """Record a failed acquisition."""
        self.failed_acquisitions += 1
    
    def record_health_check(self, success: bool) -> None:
        """Record a health check result.
        
        Args:
            success: Whether the health check passed.
        """
        self.last_health_check = datetime.now()
        if not success:
            self.health_check_failures += 1
    
    def record_timeout(self) -> None:
        """Record a timeout event."""
        self.total_timeout_events += 1
        self.last_timeout_time = datetime.now()
        self.failed_acquisitions += 1


@runtime_checkable
class IResourceProvider(Protocol):
    """Interface for resource providers."""
    
    def acquire(self, **kwargs) -> Any:
        """Acquire a resource.
        
        Args:
            **kwargs: Provider-specific parameters.
            
        Returns:
            The acquired resource.
            
        Raises:
            ResourceError: If acquisition fails.
        """
        ...
    
    def release(self, resource: Any) -> None:
        """Release a resource.
        
        Args:
            resource: The resource to release.
        """
        ...
    
    def validate(self, resource: Any) -> bool:
        """Validate that a resource is still valid.
        
        Args:
            resource: The resource to validate.
            
        Returns:
            True if the resource is valid.
        """
        ...
    
    def health_check(self) -> ResourceHealth:
        """Check the health of the resource provider.
        
        Returns:
            The health status.
        """
        ...
    
    def get_metrics(self) -> ResourceMetrics:
        """Get resource metrics.
        
        Returns:
            Current metrics.
        """
        ...


@runtime_checkable
class IResourcePool(Protocol):
    """Interface for resource pools."""
    
    def acquire(self, timeout: float | None = None) -> Any:
        """Acquire a resource from the pool.
        
        Args:
            timeout: Acquisition timeout in seconds.
            
        Returns:
            The acquired resource.
            
        Raises:
            ResourceError: If acquisition fails.
        """
        ...
    
    def release(self, resource: Any) -> None:
        """Return a resource to the pool.
        
        Args:
            resource: The resource to return.
        """
        ...
    
    def size(self) -> int:
        """Get the current pool size.
        
        Returns:
            Number of resources in the pool.
        """
        ...
    
    def available(self) -> int:
        """Get the number of available resources.
        
        Returns:
            Number of available resources.
        """
        ...
    
    def close(self) -> None:
        """Close the pool and release all resources."""
        ...


class BaseResourceProvider(ABC):
    """Base class for resource providers."""
    
    def __init__(self, name: str, config: Dict[str, Any] | None = None):
        """Initialize the provider.
        
        Args:
            name: Provider name.
            config: Provider configuration.
        """
        self.name = name
        self.config = config or {}
        self.status = ResourceStatus.IDLE
        self.metrics = ResourceMetrics()
        self._resources: List[Any] = []
    
    @abstractmethod
    def acquire(self, **kwargs) -> Any:
        """Acquire a resource.
        
        Args:
            **kwargs: Provider-specific parameters.
            
        Returns:
            The acquired resource.
        """
        pass
    
    @abstractmethod
    def release(self, resource: Any) -> None:
        """Release a resource.
        
        Args:
            resource: The resource to release.
        """
        pass
    
    def validate(self, resource: Any) -> bool:
        """Validate a resource.
        
        Args:
            resource: The resource to validate.
            
        Returns:
            True if valid.
        """
        return resource is not None
    
    def health_check(self) -> ResourceHealth:
        """Check provider health.
        
        Returns:
            Health status.
        """
        if self.status == ResourceStatus.ERROR:
            return ResourceHealth.UNHEALTHY
        elif self.status == ResourceStatus.MAINTENANCE:
            return ResourceHealth.DEGRADED
        else:
            return ResourceHealth.HEALTHY
    
    def get_metrics(self) -> ResourceMetrics:
        """Get provider metrics.
        
        Returns:
            Current metrics.
        """
        return self.metrics
    
    @contextmanager
    def resource_context(self, **kwargs):
        """Context manager for resource acquisition.
        
        Args:
            **kwargs: Acquisition parameters.
            
        Yields:
            The acquired resource.
        """
        resource = None
        start_time = datetime.now()
        try:
            resource = self.acquire(**kwargs)
            self.metrics.record_acquisition()
            yield resource
        except Exception:
            self.metrics.record_failure()
            raise
        finally:
            if resource is not None:
                hold_time = (datetime.now() - start_time).total_seconds()
                self.release(resource)
                self.metrics.record_release(hold_time)
    
    def close(self) -> None:
        """Close the provider and release all resources."""
        for resource in self._resources[:]:
            try:
                self.release(resource)
            except Exception:
                pass  # Best effort cleanup
        self._resources.clear()
        self.status = ResourceStatus.CLOSED
