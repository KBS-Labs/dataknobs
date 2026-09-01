"""Base interfaces and classes for resource management.

Teardown convention
-------------------

``ResourceManager`` holds providers of unrelated types in one registry, so the
only thing it can route teardown on is the method's *name*. The convention is
the standard one --- ``asyncio``, ``contextlib.aclosing``, and the pair
``dataknobs_common.lifecycle`` probes:

======================  ==========================================
``close()``             synchronous teardown; never a coroutine
``aclose()``            teardown that must be awaited
======================  ==========================================

A provider whose teardown must be awaited spells it ``aclose``. Spelling it
``close`` is served by the synchronous path, which calls it, discards the
coroutine it returns, and reports success --- so the teardown never runs and
nothing says otherwise. :meth:`ResourceManager.register_provider` refuses such
a provider at registration, which is the last moment its author can still act
on the mistake.

``cleanup()`` is honoured as an alternate spelling of ``aclose`` for providers
that already used it; new providers should use ``aclose``.

:class:`AsyncClosable` and :class:`AsyncCleanable` name the awaited halves, so
the routing can be a type narrowing rather than a string probe. Being
``runtime_checkable``, they test for the *presence* of the method and not for
its being a coroutine function --- that is what the registration check is for.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterator, List, Protocol, runtime_checkable
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
                self.average_acquisition_time = (self.average_acquisition_time * 0.9) + (
                    acquisition_time * 0.1
                )

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
    """Interface for resource providers.

    Teardown is *optional* and is not declared here on purpose: this Protocol
    is ``runtime_checkable``, so a new required member would change
    ``isinstance`` for every provider a consumer has already written. A
    provider that needs teardown supplies ``close()`` (synchronous) or
    ``aclose()`` (awaited) per the convention in this module's docstring;
    :meth:`ResourceManager.register_provider` enforces it, which covers
    consumer providers the Protocol cannot.
    """

    def acquire(self, **kwargs: Any) -> Any:
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

    def get_metrics(self) -> ResourceMetrics:
        """Get pool metrics.

        Returns:
            Current metrics.
        """
        ...


@runtime_checkable
class AsyncClosable(Protocol):
    """A collaborator whose teardown must be awaited.

    Named so teardown routing can narrow a type rather than probe a string.
    ``runtime_checkable`` tests only that the attribute is present --- it does
    not verify the method is a coroutine function, which is
    :meth:`ResourceManager.register_provider`'s job.
    """

    async def aclose(self) -> None:
        """Release the underlying transport."""
        ...


@runtime_checkable
class AsyncCleanable(Protocol):
    """A collaborator spelling its awaited teardown ``cleanup``.

    Honoured as an alternate spelling of :class:`AsyncClosable` for providers
    that already used it. New providers should define ``aclose``.
    """

    async def cleanup(self) -> None:
        """Release the underlying transport."""
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
    def acquire(self, **kwargs: Any) -> Any:
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
    def resource_context(self, **kwargs: Any) -> Iterator[Any]:
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
