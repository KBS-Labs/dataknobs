"""The resource-facing half of an FSM API class, written once.

:class:`~dataknobs_fsm.api.simple.SimpleFSM`,
:class:`~dataknobs_fsm.api.async_simple.AsyncSimpleFSM` and
:class:`~dataknobs_fsm.api.advanced.AdvancedFSM` are three top-level classes
with no shared base, and each grew its own resource surface. They grew
*different* ones: the two simple classes could list registered providers but
offered no way to add one, and the advanced class could add one but not list
them. A caller wanting both had to pick the class by which half it had.

This module is the shared layer that difference was evidence for. All three
methods are synchronous on every class --- registering and listing touch only
the manager's own dict, and the record is read after teardown has finished ---
so one mixin serves the async class as well as the two sync ones.

Private module, public methods: the mixin is not an extension point and is not
exported, because a consumer holds one of the three FSM classes and never this.
The leading underscore says so without hiding what it contributes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..resources.base import IResourceProvider
from ..resources.manager import ResourceManager


class ResourceSurface:
    """Register a provider, list what is registered, read what leaked.

    Mixed into all three API classes. ``_resource_manager`` is *declared* here
    and constructed by the class that mixes this in --- two of the three build
    their own manager and ``SimpleFSM`` caches the one it borrows, so the
    attribute exists on all three before any method here is reachable.
    """

    _resource_manager: ResourceManager

    def register_resource(self, name: str, resource: IResourceProvider | dict[str, Any]) -> None:
        """Register a resource provider, by instance or by configuration.

        Args:
            name: The name transforms will use to acquire this resource.
            resource: A provider instance, or a configuration dict for one --
                the dict form is handed to
                :meth:`~dataknobs_fsm.resources.manager.ResourceManager.register_from_dict`,
                which builds the provider from its ``type`` field.

        Raises:
            ResourceError: If this FSM has been closed. Teardown clears the
                registry and there is no reopen, so registering afterwards is
                always a bug and never a provider that would have been used.
            ValueError: If ``name`` is already registered, or if the provider's
                teardown method is named against its asyncness.
        """
        if isinstance(resource, dict):
            self._resource_manager.register_from_dict(name, resource)
        else:
            self._resource_manager.register_provider(name, resource)

    def get_resources(self) -> list[str]:
        """Names of the registered resource providers.

        Registered, not acquired: a name appears here from the moment its
        provider is registered, whether or not anything has asked for it.

        Read through the manager's own accessor, which copies under its lock.
        The form this replaced --- ``list(manager._providers.keys())`` --- is
        not in fact racy: ``list()`` over a dict view is a single C call that
        never yields the GIL, so unlike the Python-level sweep in ``cleanup``
        it cannot observe a concurrent write. What was wrong with it is that
        one module reached into another's private dict, which is what made
        the manager's lock discipline unauditable from the outside: an
        invariant only holds if it can be checked by reading one class.
        """
        return list(self._resource_manager.get_all_providers().keys())

    @property
    def unclosed_providers(self) -> Mapping[str, str]:
        """Providers whose teardown did not complete, name to reason.

        Empty is the normal answer, and asserting it is how a caller that
        cares about resource lifetime checks that nothing was left open::

            with SimpleFSM(config) as fsm:
                ...
            assert not fsm.unclosed_providers

        Read-only, and monotonic over the manager's life --- see
        :attr:`~dataknobs_fsm.resources.manager.ResourceManager.unclosed_providers`
        for what is recorded and why it is never cleared. The reason strings
        are diagnostic and may change; assert on the keys.

        Two populations are recorded, and **which of them a given class can
        produce depends on whether its teardown awaits**:

        =================  ===============================================
        Class              What its ``close`` can leave here
        =================  ===============================================
        ``AsyncSimpleFSM``  A provider whose teardown *raised*.
        ``SimpleFSM``       The same. Despite being the synchronous surface
                            it does not skip awaited teardown --- ``close``
                            drives the async cleanup through the shared
                            bridge, so a provider exposing ``aclose`` is
                            awaited like any other.
        ``AdvancedFSM``     Both. Its ``close`` runs the *synchronous*
                            teardown path, which cannot await, so a provider
                            exposing only ``aclose`` is skipped and named
                            here. ``aclose`` (or ``async with``) awaits it
                            instead and records nothing.
        =================  ===============================================

        So the skipped-because-unawaitable population is reachable from
        exactly one of the three, which is the case this record exists for.
        """
        return self._resource_manager.unclosed_providers
