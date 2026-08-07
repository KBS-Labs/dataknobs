"""Resource pooling implementation."""

import logging
import queue
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar

from dataknobs_common.structured_config import (
    StructuredConfig,
    StructuredConfigConsumer,
)

from dataknobs_fsm.functions.base import ResourceError
from dataknobs_fsm.resources.base import (
    IResourceProvider,
    ResourceMetrics,
)

logger = logging.getLogger(__name__)

_AT_CAPACITY: Any = object()
"""Returned by :meth:`ResourcePool._create_if_under_capacity` when the pool
may not grow.

A distinct object rather than ``None`` because ``None`` is a value a provider
may legitimately produce as the resource itself — ``IResourceProvider.acquire``
is typed ``-> Any``. Conflating the two would lose that resource after it had
already been booked into the active set, and hand the caller a timeout for a
pool that was never full.
"""


@dataclass(frozen=True)
class PoolConfig(StructuredConfig):
    """Configuration for resource pools."""
    
    min_size: int = 1
    max_size: int = 10
    acquire_timeout: float = 30.0
    idle_timeout: float = 300.0
    validation_interval: float = 60.0
    max_lifetime: float = 3600.0


@dataclass
class PooledResource:
    """A resource in a pool with metadata."""
    
    resource: Any
    created_at: datetime
    last_used: datetime
    use_count: int = 0
    
    def is_expired(self, max_lifetime: float) -> bool:
        """Check if resource has exceeded max lifetime.
        
        Args:
            max_lifetime: Maximum lifetime in seconds.
            
        Returns:
            True if expired.
        """
        age = (datetime.now() - self.created_at).total_seconds()
        return age > max_lifetime
    
    def is_idle_too_long(self, idle_timeout: float) -> bool:
        """Check if resource has been idle too long.
        
        Args:
            idle_timeout: Maximum idle time in seconds.
            
        Returns:
            True if idle too long.
        """
        idle_time = (datetime.now() - self.last_used).total_seconds()
        return idle_time > idle_timeout


class ResourcePool(StructuredConfigConsumer[PoolConfig]):
    """Thread-safe resource pool implementation.

    Built from :class:`PoolConfig` via ``StructuredConfigConsumer``. The
    required ``provider`` is a live resource provider, not config data, so it
    is supplied as an injected collaborator rather than a config field —
    mirroring the back-compat positional shortcut documented for
    ``PostgresEventBus``. Construct as ``ResourcePool(provider, config=None)``
    (the positional shortcut) or
    ``ResourcePool.from_config(config, provider=provider)``. ``self.config`` is
    the typed :class:`PoolConfig`.
    """

    CONFIG_CLS: ClassVar[type[PoolConfig]] = PoolConfig

    def __init__(
        self,
        provider: IResourceProvider,
        config: PoolConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the pool.

        Args:
            provider: Resource provider (a required live collaborator, not
                config data).
            config: Pool configuration — a typed :class:`PoolConfig`, a
                config mapping, or omitted for all-default config. Loose
                :class:`PoolConfig` field kwargs are also accepted (and may
                not be combined with a typed ``config``).
        """
        # ``provider`` travels through the mixin's collaborator channel; the
        # config (typed / dict / loose kwargs) is projected onto self.config.
        super().__init__(config, _components={"provider": provider}, **kwargs)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | StructuredConfig,
        *,
        provider: IResourceProvider,
    ) -> "ResourcePool":
        """Construct from a config dict/typed config plus the ``provider``.

        Overrides the mixin ``from_config`` so the required ``provider``
        collaborator is delivered alongside the config, routing the config
        through the inherited ``_coerce_config`` guard (a wrong
        ``StructuredConfig`` subclass raises a clear ``TypeError``).
        ``ResourcePool``'s only collaborator is the ``provider``, so no
        further ``**components`` channel is exposed.
        """
        return cls(provider, cls._coerce_config(config))

    def _setup(self) -> None:
        self.provider: IResourceProvider = self.components["provider"]
        self.metrics = ResourceMetrics()

        self._pool: queue.Queue[PooledResource] = queue.Queue(
            maxsize=self.config.max_size
        )
        self._active_resources: set = set()
        self._lock = threading.RLock()
        self._closed = False
        self._resource_map: dict = {}  # Maps resource to PooledResource

        # Initialize minimum resources
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize the pool with minimum resources."""
        for _ in range(self.config.min_size):
            try:
                resource = self.provider.acquire()
                pooled = PooledResource(
                    resource=resource,
                    created_at=datetime.now(),
                    last_used=datetime.now()
                )
                self._resource_map[id(resource)] = pooled
                self._pool.put(pooled)
            except Exception as e:
                # Continue with fewer resources, but record the failure so a
                # provider that cannot satisfy ``min_size`` is diagnosable
                # rather than silently producing an undersized pool.
                self.metrics.record_failure()
                logger.warning(
                    "Failed to pre-initialise resource for pool %s: %s",
                    self.provider.name,
                    e,
                )
    
    def acquire(self, timeout: float | None = None) -> Any:
        """Acquire a resource from the pool.

        Args:
            timeout: Maximum seconds to wait for a resource to become
                available. ``None`` (the default) uses the configured
                ``acquire_timeout``; ``0`` means do not wait at all. The wait
                applies only when the pool is at ``max_size`` — see
                :meth:`_obtain_resource`.

        Returns:
            The acquired resource.

        Raises:
            ResourceError: If acquisition fails.
        """
        if self._closed:
            raise ResourceError("Pool is closed", resource_name=self.provider.name, operation="acquire")

        # Not ``timeout or self.config.acquire_timeout``: a caller passing 0 is
        # asking not to wait, and truthiness cannot tell that from an omitted
        # argument — it would silently substitute the configured default and
        # block for it.
        timeout = self.config.acquire_timeout if timeout is None else timeout
        start_time = datetime.now()

        resource = self._obtain_resource(timeout)

        # One recording point for every way of obtaining a resource. Recording
        # it per branch is how the "took one from the pool" and "created one"
        # paths came to be counted while the "replaced an invalid one" path was
        # not.
        self.metrics.record_acquisition((datetime.now() - start_time).total_seconds())
        return resource

    def _obtain_resource(self, timeout: float) -> Any:
        """Produce a resource, waiting only when waiting is the only option.

        The pool may make a caller wait exactly when every resource it is
        allowed to create is already out on loan, because then a release by
        another holder is the only thing that can satisfy this caller. In
        every other case a resource is either sitting idle or creatable now,
        and blocking first would spend the whole ``timeout`` before reaching a
        branch that never needed to wait — a resource delivered 30 seconds
        late by default.

        Args:
            timeout: Seconds to wait once at capacity.

        Returns:
            The resource.

        Raises:
            ResourceError: If the pool is at capacity for the whole timeout.
        """
        pooled = self._take_idle()

        if pooled is None:
            resource = self._create_if_under_capacity()
            if resource is not _AT_CAPACITY:
                return resource
            # At capacity: wait for a holder to give one back.
            pooled = self._take_idle(timeout=timeout)

        if pooled is None:
            # Capacity can be freed *without* anything reaching the queue: a
            # resource past its max lifetime is retired on release rather than
            # returned, so nothing wakes a waiter. Re-check before giving up.
            resource = self._create_if_under_capacity()
            if resource is not _AT_CAPACITY:
                return resource

            self.metrics.record_timeout()
            raise ResourceError(
                f"Failed to acquire resource within {timeout} seconds",
                resource_name=self.provider.name,
                operation="acquire"
            )

        if not self._validate_pooled_resource(pooled):
            # Resource is invalid, create a new one
            self._release_pooled_resource(pooled)
            return self._create_new_resource()

        # Update metadata
        pooled.last_used = datetime.now()
        pooled.use_count += 1

        with self._lock:
            self._active_resources.add(id(pooled.resource))

        return pooled.resource

    def _take_idle(self, timeout: float | None = None) -> PooledResource | None:
        """Take an idle resource from the queue.

        Args:
            timeout: Seconds to wait for one. ``None`` (the default) polls
                without blocking.

        Returns:
            The pooled resource, or ``None`` if none became available.
        """
        try:
            if timeout is None:
                return self._pool.get_nowait()
            return self._pool.get(timeout=timeout)
        except queue.Empty:
            return None

    def _create_if_under_capacity(self) -> Any:
        """Create a resource if the pool is below ``max_size``.

        The check and the creation share one hold of the lock, so two callers
        arriving at the last free slot cannot both pass the check and
        over-allocate past ``max_size``.

        Returns:
            The new resource, or :data:`_AT_CAPACITY` if the pool may not
            grow. Sentinel rather than ``None``; see that constant.

        Raises:
            ResourceError: If the provider fails to produce a resource.
        """
        with self._lock:
            if len(self._active_resources) >= self.config.max_size:
                return _AT_CAPACITY
            return self._create_new_resource()


    def release(self, resource: Any) -> None:
        """Return a resource to the pool.
        
        Args:
            resource: The resource to return.
        """
        if self._closed:
            # Pool is closed, just release the resource
            self.provider.release(resource)
            return
        
        resource_id = id(resource)
        
        with self._lock:
            if resource_id not in self._active_resources:
                return  # Resource not from this pool
            
            self._active_resources.discard(resource_id)
            pooled = self._resource_map.get(resource_id)
            
            if pooled is None:
                # Create new pooled resource wrapper
                pooled = PooledResource(
                    resource=resource,
                    created_at=datetime.now(),
                    last_used=datetime.now()
                )
                self._resource_map[resource_id] = pooled
            
            # Check if resource should be retired
            if (pooled.is_expired(self.config.max_lifetime) or
                not self.provider.validate(resource)):
                self._release_pooled_resource(pooled)
            else:
                # Return to pool
                pooled.last_used = datetime.now()
                try:
                    self._pool.put_nowait(pooled)
                except queue.Full:
                    # Pool is full, release the resource
                    self._release_pooled_resource(pooled)
        
        hold_time = 0.1  # Default hold time
        self.metrics.record_release(hold_time)
    
    def _validate_pooled_resource(self, pooled: PooledResource) -> bool:
        """Validate a pooled resource.
        
        Args:
            pooled: The pooled resource to validate.
            
        Returns:
            True if valid.
        """
        if pooled.is_expired(self.config.max_lifetime):
            return False
        
        if pooled.is_idle_too_long(self.config.idle_timeout):
            return False
        
        return self.provider.validate(pooled.resource)
    
    def _create_new_resource(self) -> Any:
        """Create a new resource.
        
        Returns:
            The new resource.
            
        Raises:
            ResourceError: If creation fails.
        """
        try:
            resource = self.provider.acquire()
            pooled = PooledResource(
                resource=resource,
                created_at=datetime.now(),
                last_used=datetime.now(),
                use_count=1
            )
            
            with self._lock:
                self._resource_map[id(resource)] = pooled
                self._active_resources.add(id(resource))
            
            # Note: Acquisition metrics are now tracked in acquire() method with timing
            return resource
            
        except Exception as e:
            self.metrics.record_failure()
            raise ResourceError(
                f"Failed to create resource ({type(e).__name__})",
                resource_name=self.provider.name,
                operation="create"
            ) from e
    
    def _release_pooled_resource(self, pooled: PooledResource) -> None:
        """Release a pooled resource.
        
        Args:
            pooled: The pooled resource to release.
        """
        resource_id = id(pooled.resource)
        try:
            # Attempt to properly release the resource
            self.provider.release(pooled.resource)
            logger.debug(
                "Successfully released pooled resource %s from pool %s",
                resource_id,
                self.provider.name,
            )
        except AttributeError as e:
            logger.warning(
                "Resource provider %s missing release method: %s",
                self.provider.name,
                e,
            )
        except Exception as e:
            logger.error(
                "Error releasing pooled resource %s from pool %s: %s",
                resource_id,
                self.provider.name,
                e,
            )
            # Track the error for debugging
            self.metrics.record_failure()
            # Still remove from map even if release failed to prevent memory leak
        
        with self._lock:
            self._resource_map.pop(resource_id, None)
    
    def size(self) -> int:
        """Get the current pool size.
        
        Returns:
            Total number of resources (active + idle).
        """
        with self._lock:
            return self._pool.qsize() + len(self._active_resources)
    
    def available(self) -> int:
        """Get the number of available resources.
        
        Returns:
            Number of idle resources in pool.
        """
        return self._pool.qsize()
    
    def close(self) -> None:
        """Close the pool and release all resources."""
        self._closed = True
        
        # Release active resources
        with self._lock:
            for resource_id in list(self._active_resources):
                pooled = self._resource_map.get(resource_id)
                if pooled:
                    self._release_pooled_resource(pooled)
            self._active_resources.clear()
        
        # Release pooled resources
        while not self._pool.empty():
            try:
                pooled = self._pool.get_nowait()
                self._release_pooled_resource(pooled)
            except queue.Empty:
                break
        
        self._resource_map.clear()
    
    def evict_idle(self) -> int:
        """Evict idle resources that have exceeded timeout.
        
        Returns:
            Number of resources evicted.
        """
        evicted = 0
        temp_resources = []
        
        # Check all pooled resources
        while not self._pool.empty():
            try:
                pooled = self._pool.get_nowait()
                if pooled.is_idle_too_long(self.config.idle_timeout):
                    self._release_pooled_resource(pooled)
                    evicted += 1
                else:
                    temp_resources.append(pooled)
            except queue.Empty:
                break
        
        # Put back the non-evicted resources
        for pooled in temp_resources:
            try:
                self._pool.put_nowait(pooled)
            except queue.Full:
                self._release_pooled_resource(pooled)
        
        return evicted
    
    def get_metrics(self) -> ResourceMetrics:
        """Get pool metrics.
        
        Returns:
            Current metrics.
        """
        return self.metrics
