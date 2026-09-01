"""Central resource manager for FSM.

Teardown is routed by method name --- see the convention in
:mod:`dataknobs_fsm.resources.base`. This module both enforces it
(:meth:`ResourceManager.register_provider`) and reports where it cannot be
honoured (:attr:`ResourceManager.unclosed_providers`).
"""

import asyncio
import logging
import threading
from contextlib import contextmanager
from types import MappingProxyType, TracebackType
from typing import Any, Dict, Iterator, Mapping, Self, Set

from dataknobs_common import is_async_callable

from dataknobs_fsm.functions.base import ResourceError, ResourceConfig
from dataknobs_fsm.resources.base import (
    AsyncCleanable,
    AsyncClosable,
    IResourceProvider,
    IResourcePool,
    ResourceStatus,
    ResourceHealth,
    ResourceMetrics,
)
from dataknobs_fsm.resources.pool import ResourcePool, PoolConfig

logger = logging.getLogger(__name__)

#: Recorded against a provider whose awaited teardown the synchronous path
#: cannot run. The text is diagnostic, not contractual --- assert on the keys
#: of :attr:`ResourceManager.unclosed_providers`, never on these strings.
_SKIPPED_ASYNC_TEARDOWN = (
    "teardown must be awaited (aclose); close() cannot run it, "
    "so the underlying transport is still open"
)


class ResourceManager:
    """Manages resources across the FSM system."""

    def __init__(self) -> None:
        """Initialize the resource manager."""
        self._providers: Dict[str, IResourceProvider] = {}
        self._pools: Dict[str, IResourcePool] = {}
        self._resources: Dict[str, Any] = {}
        self._resource_owners: Dict[str, Set[str]] = {}  # resource_name -> owner_ids
        self._lock = threading.RLock()
        self._closed = False
        self._unclosed_providers: Dict[str, str] = {}

    @property
    def unclosed_providers(self) -> Mapping[str, str]:
        """Providers whose teardown did not complete, name to reason.

        Empty is the normal answer, and asserting it is how a caller that
        cares about resource lifetime checks that nothing was left open::

            with SimpleFSM(config) as fsm:
                ...
            assert not fsm.unclosed_providers

        Two populations are recorded: a provider whose teardown must be
        awaited but was closed synchronously, and a provider whose teardown
        raised. A provider exposing no teardown at all is *not* recorded ---
        there was nothing to close, which is a legitimate shape.

        Lifetime-scoped and monotonic: entries accumulate and are never
        cleared. :meth:`close` is terminal and empties the registry, so a
        second call finds nothing to close and would otherwise reset the
        record, erasing the first call's evidence.

        The reason strings are diagnostic and may change. Assert on the keys.
        """
        with self._lock:
            return MappingProxyType(dict(self._unclosed_providers))

    def _record_unclosed(self, name: str, reason: str) -> None:
        """Note that ``name``'s teardown did not complete."""
        with self._lock:
            self._unclosed_providers[name] = reason

    def register_provider(
        self, name: str, provider: IResourceProvider, pool_config: PoolConfig | None = None
    ) -> None:
        """Register a resource provider.

        Args:
            name: Resource name.
            provider: Resource provider.
            pool_config: Optional pool configuration.

        Raises:
            ValueError: If ``name`` is taken, or if the provider spells an
                awaited teardown ``close`` (see the teardown convention in
                :mod:`dataknobs_fsm.resources.base`).
        """
        # Every provider enters here --- `register_from_dict` and the config
        # builder both end at this call --- so it is the one place the teardown
        # convention can be enforced for providers this package never sees.
        # Raising rather than warning: a provider whose `close` must be awaited
        # cannot be torn down correctly by any caller of this manager, and this
        # is the last moment its author can still act on the mistake.
        if is_async_callable(getattr(provider, "close", None)):
            raise ValueError(
                f"Provider '{name}' defines an async close(). Resource teardown "
                "is synchronous by convention; name an awaitable teardown "
                "'aclose()' so ResourceManager.cleanup() can await it."
            )

        with self._lock:
            if name in self._providers:
                raise ValueError(f"Provider '{name}' already registered")

            self._providers[name] = provider

            if pool_config:
                # Create a pool for this provider
                pool = ResourcePool(provider, pool_config)
                self._pools[name] = pool

    def unregister_provider(self, name: str) -> None:
        """Unregister a resource provider.

        Args:
            name: Resource name.
        """
        with self._lock:
            # Close pool if exists
            if name in self._pools:
                self._pools[name].close()
                del self._pools[name]

            # Remove provider
            if name in self._providers:
                provider = self._providers[name]
                if hasattr(provider, "close"):
                    provider.close()
                del self._providers[name]

    def get_provider(self, name: str) -> IResourceProvider | None:
        """Get a resource provider by name.

        Args:
            name: Resource name.

        Returns:
            The resource provider or None if not found.
        """
        with self._lock:
            return self._providers.get(name)

    def get_all_providers(self) -> Dict[str, IResourceProvider]:
        """Get all registered resource providers.

        Returns:
            Dictionary of resource name to provider.
        """
        with self._lock:
            return dict(self._providers)

    def acquire(self, name: str, owner_id: str, timeout: float | None = None, **kwargs: Any) -> Any:
        """Acquire a resource.

        Args:
            name: Resource name.
            owner_id: ID of the owner (e.g., state instance ID).
            timeout: Acquisition timeout.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The acquired resource.

        Raises:
            ResourceError: If acquisition fails.
        """
        if self._closed:
            raise ResourceError(
                "Resource manager is closed", resource_name=name, operation="acquire"
            )

        # First check if provider exists and if owner already has resource
        with self._lock:
            if name not in self._providers:
                raise ResourceError(
                    f"Unknown resource '{name}'", resource_name=name, operation="acquire"
                )

            # Check if owner already has this resource
            owner_key = f"{owner_id}:{name}"
            if owner_key in self._resources:
                return self._resources[owner_key]

            # Check if we have a pool for this resource
            has_pool = name in self._pools

        # Acquire resource outside of lock to prevent deadlock
        if has_pool:
            resource = self._pools[name].acquire(timeout)
        else:
            resource = self._providers[name].acquire(**kwargs)

        # Re-acquire lock to track ownership
        with self._lock:
            # Double-check that owner doesn't have resource (race condition check)
            owner_key = f"{owner_id}:{name}"
            if owner_key in self._resources:
                # Another thread already acquired for this owner, release the extra
                if has_pool:
                    self._pools[name].release(resource)
                else:
                    self._providers[name].release(resource)
                return self._resources[owner_key]

            # Track ownership
            self._resources[owner_key] = resource
            if name not in self._resource_owners:
                self._resource_owners[name] = set()
            self._resource_owners[name].add(owner_id)

            return resource

    def release(self, name: str, owner_id: str) -> None:
        """Release a resource.

        Args:
            name: Resource name.
            owner_id: ID of the owner.
        """
        with self._lock:
            owner_key = f"{owner_id}:{name}"

            if owner_key not in self._resources:
                return  # Resource not acquired or already released

            resource = self._resources[owner_key]

            # Release to pool or provider
            if name in self._pools:
                self._pools[name].release(resource)
            elif name in self._providers:
                self._providers[name].release(resource)

            # Clean up tracking
            del self._resources[owner_key]
            if name in self._resource_owners:
                self._resource_owners[name].discard(owner_id)

    def release_all(self, owner_id: str) -> None:
        """Release all resources owned by an owner.

        Args:
            owner_id: ID of the owner.
        """
        with self._lock:
            # Find all resources owned by this owner
            owner_resources = []
            for key in list(self._resources.keys()):
                if key.startswith(f"{owner_id}:"):
                    resource_name = key.split(":", 1)[1]
                    owner_resources.append(resource_name)

            # Release them all
            for resource_name in owner_resources:
                self.release(resource_name, owner_id)

    def get_resource(self, name: str, owner_id: str) -> Any | None:
        """Get an acquired resource.

        Args:
            name: Resource name.
            owner_id: ID of the owner.

        Returns:
            The resource if acquired, None otherwise.
        """
        owner_key = f"{owner_id}:{name}"
        return self._resources.get(owner_key)

    def has_resource(self, name: str, owner_id: str) -> bool:
        """Check if an owner has acquired a resource.

        Args:
            name: Resource name.
            owner_id: ID of the owner.

        Returns:
            True if the owner has the resource.
        """
        owner_key = f"{owner_id}:{name}"
        return owner_key in self._resources

    def validate_resource(self, name: str) -> bool:
        """Validate a resource provider.

        Args:
            name: Resource name.

        Returns:
            True if the resource is valid.
        """
        with self._lock:
            if name not in self._providers:
                return False

            # Create a test resource to validate
            try:
                resource = self._providers[name].acquire()
                valid = self._providers[name].validate(resource)
                self._providers[name].release(resource)
                return valid
            except Exception:
                return False

    def health_check(self, name: str | None = None) -> Dict[str, ResourceHealth]:
        """Check health of resources.

        Args:
            name: Optional specific resource name.

        Returns:
            Health status by resource name.
        """
        with self._lock:
            if name:
                if name in self._providers:
                    return {name: self._providers[name].health_check()}
                else:
                    return {name: ResourceHealth.UNKNOWN}

            # Check all resources
            health_status = {}
            for resource_name, provider in self._providers.items():
                try:
                    health_status[resource_name] = provider.health_check()
                except Exception:
                    health_status[resource_name] = ResourceHealth.UNKNOWN

            return health_status

    def get_metrics(self, name: str | None = None) -> Dict[str, ResourceMetrics]:
        """Get resource metrics.

        Args:
            name: Optional specific resource name.

        Returns:
            Metrics by resource name.
        """
        with self._lock:
            if name:
                metrics = {}
                if name in self._providers:
                    metrics[name] = self._providers[name].get_metrics()
                if name in self._pools:
                    metrics[f"{name}_pool"] = self._pools[name].get_metrics()
                return metrics

            # Get all metrics
            all_metrics = {}
            for resource_name, provider in self._providers.items():
                all_metrics[resource_name] = provider.get_metrics()
            for resource_name, pool in self._pools.items():
                all_metrics[f"{resource_name}_pool"] = pool.get_metrics()

            return all_metrics

    @contextmanager
    def resource_context(
        self, name: str, owner_id: str, timeout: float | None = None, **kwargs: Any
    ) -> Iterator[Any]:
        """Context manager for resource acquisition.

        Args:
            name: Resource name.
            owner_id: ID of the owner.
            timeout: Acquisition timeout.
            **kwargs: Additional parameters.

        Yields:
            The acquired resource.
        """
        resource = self.acquire(name, owner_id, timeout, **kwargs)
        try:
            yield resource
        finally:
            self.release(name, owner_id)

    def configure_from_requirements(
        self, requirements: list[ResourceConfig], owner_id: str
    ) -> Dict[str, Any]:
        """Configure resources from requirements.

        Args:
            requirements: List of resource configurations.
            owner_id: ID of the owner.

        Returns:
            Dictionary of acquired resources.

        Raises:
            ResourceError: If any resource cannot be acquired.
        """
        acquired = {}

        try:
            for config in requirements:
                resource = self.acquire(config.name, owner_id, timeout=config.timeout)
                acquired[config.name] = resource

            return acquired

        except Exception as e:
            # Release any acquired resources on failure
            for name in acquired:
                try:
                    self.release(name, owner_id)
                except Exception:
                    pass
            raise ResourceError(
                f"Failed to acquire resources ({type(e).__name__})",
                resource_name="multiple",
                operation="configure",
            ) from e

    def _release_acquired_and_close_pools(self) -> None:
        """Return every acquired resource and close every pool.

        Shared by :meth:`close` and :meth:`cleanup` so the two halves of the
        lifecycle cannot drift. They already had: the async half cleared
        ``_pools`` without closing the pools, so a pool's resources were
        never handed back to their provider and its connections stayed open
        until garbage collection — while the sync half released them
        correctly.

        Ordering is load-bearing: resources are released *before* providers
        are closed, since releasing hands them back to the provider that
        issued them.
        """
        with self._lock:
            # Release all acquired resources
            for owner_id in {key.split(":")[0] for key in self._resources.keys()}:
                self.release_all(owner_id)

            # Close all pools
            for pool in self._pools.values():
                pool.close()
            self._pools.clear()

    def close(self) -> None:
        """Close the resource manager and release all resources.

        Terminal: a closed manager rejects further :meth:`acquire` calls.
        Use :meth:`cleanup` from async code — it does everything this does
        and additionally awaits providers whose cleanup is a coroutine.

        Never raises: a provider that fails teardown is recorded in
        :attr:`unclosed_providers` and the remaining providers are still
        closed. This method is reachable from ``__exit__``, where propagating
        would replace whatever the ``with`` body was raising.
        """
        self._closed = True
        self._release_acquired_and_close_pools()

        with self._lock:
            # Close all providers. Per-provider isolation matches `cleanup`:
            # without it one raising provider abandons every provider after it
            # in iteration order and skips the registry clear below, leaving a
            # manager marked closed while still holding everything it failed to
            # close. `close` is reachable from `__exit__`, where propagating
            # would additionally replace whatever the `with` body was raising --
            # so the failure is recorded instead of thrown.
            for name, provider in self._providers.items():
                self._close_provider(name, provider)
            self._providers.clear()

            self._resources.clear()
            self._resource_owners.clear()

    async def cleanup(self) -> None:
        """Async cleanup of all resource providers.

        This method performs async cleanup of resources that support it,
        while falling back to sync cleanup for those that don't.

        A strict superset of :meth:`close`: it releases the same acquired
        resources, closes the same pools, and leaves the manager equally
        closed — then additionally awaits providers exposing an ``aclose`` /
        ``cleanup`` coroutine. That is what makes "prefer ``aclose()`` from
        async code" safe advice; while this skipped the release, the pool
        close, and the closed flag, the async form was in some ways the
        *weaker* one, and the two even reported a later ``acquire`` failure
        with different messages for the same underlying state.
        """
        # Claim closure up front exactly as `close` does, so the terminal
        # state does not depend on which half the caller reached for.
        self._closed = True

        # Off the event loop: releasing resources and closing pools calls
        # into provider code that may block.
        await asyncio.to_thread(self._release_acquired_and_close_pools)

        # Names are carried alongside the coroutines: `gather` returns results
        # positionally, so without the pairing a failure can only be reported
        # as an index into a list the reader cannot see.
        awaited: list[tuple[str, Any]] = []
        sync_providers: list[tuple[str, IResourceProvider]] = []

        # Separate async and sync providers
        for name, provider in self._providers.items():
            if isinstance(provider, AsyncClosable):
                awaited.append((name, self._async_close_provider(name, provider)))
            elif isinstance(provider, AsyncCleanable):
                awaited.append((name, self._async_cleanup_provider(name, provider)))
            else:
                # Provider only has sync cleanup
                sync_providers.append((name, provider))

        # Run async cleanups concurrently
        if awaited:
            results = await asyncio.gather(*(task for _, task in awaited), return_exceptions=True)
            for (name, _), result in zip(awaited, results, strict=True):
                if isinstance(result, BaseException):
                    logger.error("Error during async cleanup of %s: %s", name, result)
                    self._record_unclosed(name, f"aclose() raised: {result}")

        # Run sync cleanups in executor to avoid blocking
        if sync_providers:
            loop = asyncio.get_running_loop()
            for name, provider in sync_providers:
                try:
                    await loop.run_in_executor(None, self._close_provider, name, provider)
                except Exception as e:
                    logger.error("Error closing sync provider %s: %s", name, e)
                    self._record_unclosed(name, f"close() raised: {e}")

        # Clear tracking data
        with self._lock:
            self._resources.clear()
            self._resource_owners.clear()
            self._pools.clear()
            self._providers.clear()

    async def _async_close_provider(self, name: str, provider: AsyncClosable) -> None:
        """Close a provider whose teardown must be awaited.

        Args:
            name: Provider name
            provider: Provider instance
        """
        try:
            await provider.aclose()
            logger.debug("Successfully closed async provider %s", name)
        except Exception as e:
            logger.error("Error closing async provider %s: %s", name, e)
            raise

    async def _async_cleanup_provider(self, name: str, provider: AsyncCleanable) -> None:
        """Clean up a provider spelling its awaited teardown ``cleanup``.

        Args:
            name: Provider name
            provider: Provider instance
        """
        try:
            await provider.cleanup()
            logger.debug("Successfully cleaned up async provider %s", name)
        except Exception as e:
            logger.error("Error cleaning up async provider %s: %s", name, e)
            raise

    def _close_provider(self, name: str, provider: IResourceProvider) -> None:
        """Close a provider synchronously, recording what it could not finish.

        Two things can go wrong here and neither may propagate --- this runs
        from :meth:`close`, which is reachable from ``__exit__`` --- so both
        are recorded in :attr:`unclosed_providers` instead.

        The provider's awaited teardown cannot be run from here: there is no
        loop to await it on, and starting one is not available to a method
        ``__exit__`` may call while a loop is already running in this thread.
        Its synchronous ``close`` still runs (on a
        :class:`~dataknobs_fsm.resources.base.BaseResourceProvider` that
        releases the acquired handles), and what it could not do is reported
        rather than logged as a success.

        Args:
            name: Provider name
            provider: Provider instance
        """
        needs_await = isinstance(provider, (AsyncClosable, AsyncCleanable))
        close = getattr(provider, "close", None)

        if close is not None:
            try:
                close()
            except Exception as e:
                logger.error("Error closing provider %s: %s", name, e)
                self._record_unclosed(name, f"close() raised: {e}")
                return

        if needs_await:
            logger.error(
                "Provider %s was closed synchronously; its awaited teardown was "
                "NOT run and the underlying transport is still open. Use "
                "`await cleanup()`, or close the FSM with `await aclose()` / "
                "`async with`.",
                name,
            )
            self._record_unclosed(name, _SKIPPED_ASYNC_TEARDOWN)
        elif close is not None:
            logger.debug("Successfully closed sync provider %s", name)

    def create_provider_from_dict(self, name: str, config: Dict[str, Any]) -> IResourceProvider:
        """Create a resource provider from a dictionary configuration.

        Args:
            name: Resource name
            config: Dictionary configuration for the resource

        Returns:
            Resource provider instance
        """

        # Create a simple in-memory resource provider
        class SimpleResourceProvider(IResourceProvider):
            """Simple in-memory resource provider for testing and basic use cases.

            Provides a lightweight resource provider that stores data in memory
            from configuration. Useful for testing FSMs without external dependencies
            or for simple static data resources.
            """

            def __init__(self, name: str, config: Dict[str, Any]) -> None:
                self.name = name
                self.config = config
                self.data = config.get("data", {})
                self._status = ResourceStatus.IDLE

            def acquire(self, **kwargs: Any) -> Any:
                self._status = ResourceStatus.BUSY
                return self.data

            def release(self, resource: Any) -> None:
                self._status = ResourceStatus.IDLE

            def validate(self, resource: Any) -> bool:
                return resource is not None

            def health_check(self) -> ResourceHealth:
                return ResourceHealth.HEALTHY

            def get_metrics(self) -> ResourceMetrics:
                return ResourceMetrics(
                    total_acquisitions=0,
                    active_connections=1 if self._status == ResourceStatus.BUSY else 0,
                    failed_acquisitions=0,
                )

            async def get_resource(self) -> Any:
                return self.data

            def close(self) -> None:
                """Nothing to release: the data is a dict from the config."""

        return SimpleResourceProvider(name, config)

    def create_simple_provider(self, name: str, data: Any) -> IResourceProvider:
        """Create a simple resource provider with static data.

        Args:
            name: Resource name
            data: The resource data to provide

        Returns:
            Resource provider instance
        """
        return self.create_provider_from_dict(name, {"data": data})

    def register_from_dict(self, name: str, config: Dict[str, Any]) -> None:
        """Register a resource provider from a dictionary configuration.

        Args:
            name: Resource name
            config: Dictionary configuration for the resource
        """
        provider = self.create_provider_from_dict(name, config)
        self.register_provider(name, provider)

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self.close()

    def get_resource_status(self, name: str) -> Dict[str, Any]:
        """Get status information for a resource.

        Args:
            name: Resource name.

        Returns:
            Status dictionary with provider and pool information.
        """
        with self._lock:
            status = {
                "provider_exists": name in self._providers,
                "has_pool": name in self._pools,
                "active_count": 0,
                "owners": list(self._resource_owners.get(name, set())),
            }

            if name in self._providers:
                try:
                    metrics = self._providers[name].get_metrics()
                    status["active_count"] = metrics.active_connections
                    status["total_acquires"] = metrics.total_acquisitions
                    status["total_releases"] = (
                        metrics.total_acquisitions - metrics.active_connections
                    )
                except Exception:
                    pass

            return status

    def get_all_resources(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered resources.

        Returns:
            Dictionary mapping resource names to their status.
        """
        with self._lock:
            all_resources = {}
            for name in self._providers:
                all_resources[name] = self.get_resource_status(name)
            return all_resources

    def get_resource_owners(self, name: str) -> Set[str]:
        """Get all owners of a specific resource.

        Args:
            name: Resource name.

        Returns:
            Set of owner IDs.
        """
        with self._lock:
            return self._resource_owners.get(name, set()).copy()
