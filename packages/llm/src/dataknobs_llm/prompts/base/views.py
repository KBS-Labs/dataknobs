# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Two named doors between the two prompt-library flavours.

A consumer holds one flavour and a library is the other. That happens in both
directions and the two directions are **not** the same price, which is the
whole reason these are named functions rather than a single ``adapt()``:

======================  =========================================================
``as_async(library)``   A synchronous library presented to an async consumer.
                        The call is offloaded to a worker thread, so a library
                        that reads a file to answer cannot stall the consumer's
                        loop. No thread is owned, nothing is held open, and
                        there is nothing to close.
``as_sync(library)``    An asynchronous library presented to a synchronous
                        consumer. Costs a private event loop on a daemon
                        thread, held until :meth:`SyncPromptLibraryView.close`,
                        and **blocks the calling thread** for the whole of
                        every call.
======================  =========================================================

The asymmetry is the lesson. Wrapping sync-for-async is an accommodation;
wrapping async-for-sync is a bridge, and a bridge has a lifetime, a thread and
a failure mode. Where a consumer can take the async library directly, that is
always the better answer than ``as_sync``.

**Do not hand an** ``as_sync`` **result to** :class:`AsyncPromptBuilder`. That
builder calls its library synchronously from inside its own ``async def``, so
every template fetch would block the builder's event loop for the wrapped
library's round trip --- which is exactly the harm ``as_sync`` looks like it is
solving. Give it the async library directly once it accepts one; until then, a
versioned library's own ``await get_version(...)`` is the route that costs
nothing.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from dataknobs_common.sync_bridge import SyncBridgeAdapter

from .abstract_prompt_library import AbstractPromptLibrary
from .async_prompt_library import AsyncPromptLibrary

if TYPE_CHECKING:
    from dataknobs_common.sync_bridge import SyncLoopBridge

    from .types import MessageIndex, PromptTemplateDict, RAGConfig

__all__ = [
    "AsyncPromptLibraryView",
    "SyncPromptLibraryView",
    "as_async",
    "as_sync",
]


class AsyncPromptLibraryView(AsyncPromptLibrary):
    """A synchronous library, reached by an asynchronous consumer.

    Every reaching member is forwarded through :func:`asyncio.to_thread`, so
    the consumer's loop is free for the duration of the call. That is not
    ceremony over an in-memory dictionary: this view is handed *any*
    :class:`AbstractPromptLibrary`, including one that opens a file or a socket
    to answer, and a shipped ``async def`` cannot see who else is on its loop.
    The three libraries shipped here answer from memory, so the hop is the
    whole cost there.

    :meth:`get_metadata` is not forwarded through a thread: it is synchronous
    on both halves precisely because it reaches nothing.

    Nothing is owned and nothing is held, so there is no ``close()`` and no
    teardown obligation --- the wrapped library's lifetime is still the
    caller's business, exactly as it was before the view existed.
    """

    def __init__(self, library: AbstractPromptLibrary) -> None:
        """Args:
        library: The synchronous library to present asynchronously. Not this
            view's to close; it was built elsewhere.
        """
        self._library = library

    @property
    def library(self) -> AbstractPromptLibrary:
        """The wrapped library --- a view does not reshape."""
        return self._library

    async def get_system_prompt(self, name: str, **kwargs: Any) -> PromptTemplateDict | None:
        """Awaitable :meth:`AbstractPromptLibrary.get_system_prompt`."""
        return await asyncio.to_thread(self._library.get_system_prompt, name, **kwargs)

    async def list_system_prompts(self) -> list[str]:
        """Awaitable :meth:`AbstractPromptLibrary.list_system_prompts`."""
        return await asyncio.to_thread(self._library.list_system_prompts)

    async def get_user_prompt(self, name: str, **kwargs: Any) -> PromptTemplateDict | None:
        """Awaitable :meth:`AbstractPromptLibrary.get_user_prompt`."""
        return await asyncio.to_thread(self._library.get_user_prompt, name, **kwargs)

    async def list_user_prompts(self) -> list[str]:
        """Awaitable :meth:`AbstractPromptLibrary.list_user_prompts`."""
        return await asyncio.to_thread(self._library.list_user_prompts)

    async def get_message_index(self, name: str, **kwargs: Any) -> MessageIndex | None:
        """Awaitable :meth:`AbstractPromptLibrary.get_message_index`."""
        return await asyncio.to_thread(self._library.get_message_index, name, **kwargs)

    async def list_message_indexes(self) -> list[str]:
        """Awaitable :meth:`AbstractPromptLibrary.list_message_indexes`."""
        return await asyncio.to_thread(self._library.list_message_indexes)

    async def get_rag_config(self, name: str, **kwargs: Any) -> RAGConfig | None:
        """Awaitable :meth:`AbstractPromptLibrary.get_rag_config`."""
        return await asyncio.to_thread(self._library.get_rag_config, name, **kwargs)

    async def get_prompt_rag_configs(
        self, prompt_name: str, prompt_type: str = "user", **kwargs: Any
    ) -> list[RAGConfig]:
        """Awaitable :meth:`AbstractPromptLibrary.get_prompt_rag_configs`."""
        return await asyncio.to_thread(
            self._library.get_prompt_rag_configs, prompt_name, prompt_type, **kwargs
        )

    def get_metadata(self) -> dict[str, Any]:
        """The wrapped library's metadata, unchanged and unwrapped."""
        return self._library.get_metadata()

    async def reload(self) -> None:
        """Awaitable :meth:`AbstractPromptLibrary.reload`.

        Offloaded like the getters, and with more reason: a reload is the
        member most likely to re-read whatever the library is built over.
        """
        await asyncio.to_thread(self._library.reload)

    async def validate(self) -> list[str]:
        """Awaitable :meth:`AbstractPromptLibrary.validate`."""
        return await asyncio.to_thread(self._library.validate)


class SyncPromptLibraryView(SyncBridgeAdapter, AbstractPromptLibrary):
    """An asynchronous library, reached by a synchronous consumer.

    A :class:`~dataknobs_common.sync_bridge.SyncBridgeAdapter`, so every call
    runs on a private event loop on a daemon thread. That makes the library
    reachable from plain ``def`` code *and* from inside a running loop without
    the ``run_until_complete`` deadlock.

    "Callable from inside a running loop" means it does not *deadlock*. It
    still **blocks**: the calling thread waits on the bridge for the whole
    call, so if that thread is running an event loop, every other task on it is
    stalled for a database round trip. From async code, await the library
    directly --- this class is for the ``def`` sites that cannot.

    ```python
    with as_sync(versioned_library, timeout=30) as library:
        template = library.get_system_prompt("greeting")
    ```

    The bridge is built on first use and ended by :meth:`close`, for which
    ``with`` is the context-manager form; an async holder uses ``async with``
    and :meth:`aclose` so its own loop is not blocked for the teardown. Hand
    several views the same ``bridge`` and they share the one thread.

    The library handed in is **not** this view's to close. It was built
    elsewhere, so ``close()`` ends only the bridge --- which is why no
    ``_close_inner`` is overridden here.
    """

    BRIDGE_THREAD_NAME = "dk-sync-prompt-library"

    def __init__(
        self,
        library: AsyncPromptLibrary,
        *,
        bridge: SyncLoopBridge | None = None,
        timeout: float | None = None,
    ) -> None:
        """Args:
        library: The asynchronous library to reach.
        bridge: A bridge to run this library's coroutines on. The default
            builds a private one on first use and ends it in ``close()``.
        timeout: Seconds to allow each call, giving a synchronous caller an
            upper bound on a blocking wait it cannot otherwise cancel.
        """
        super().__init__(bridge=bridge, timeout=timeout)
        self._library = library

    @property
    def library(self) -> AsyncPromptLibrary:
        """The wrapped library --- a view does not reshape."""
        return self._library

    def get_system_prompt(self, name: str, **kwargs: Any) -> PromptTemplateDict | None:
        """Blocking :meth:`AsyncPromptLibrary.get_system_prompt`."""
        return self._run(self._library.get_system_prompt(name, **kwargs))

    def list_system_prompts(self) -> list[str]:
        """Blocking :meth:`AsyncPromptLibrary.list_system_prompts`."""
        return self._run(self._library.list_system_prompts())

    def get_user_prompt(self, name: str, **kwargs: Any) -> PromptTemplateDict | None:
        """Blocking :meth:`AsyncPromptLibrary.get_user_prompt`."""
        return self._run(self._library.get_user_prompt(name, **kwargs))

    def list_user_prompts(self) -> list[str]:
        """Blocking :meth:`AsyncPromptLibrary.list_user_prompts`."""
        return self._run(self._library.list_user_prompts())

    def get_message_index(self, name: str, **kwargs: Any) -> MessageIndex | None:
        """Blocking :meth:`AsyncPromptLibrary.get_message_index`."""
        return self._run(self._library.get_message_index(name, **kwargs))

    def list_message_indexes(self) -> list[str]:
        """Blocking :meth:`AsyncPromptLibrary.list_message_indexes`."""
        return self._run(self._library.list_message_indexes())

    def get_rag_config(self, name: str, **kwargs: Any) -> RAGConfig | None:
        """Blocking :meth:`AsyncPromptLibrary.get_rag_config`."""
        return self._run(self._library.get_rag_config(name, **kwargs))

    def get_prompt_rag_configs(
        self, prompt_name: str, prompt_type: str = "user", **kwargs: Any
    ) -> list[RAGConfig]:
        """Blocking :meth:`AsyncPromptLibrary.get_prompt_rag_configs`."""
        return self._run(self._library.get_prompt_rag_configs(prompt_name, prompt_type, **kwargs))

    def get_metadata(self) -> dict[str, Any]:
        """The wrapped library's metadata, read without touching the bridge."""
        return self._library.get_metadata()

    def reload(self) -> None:
        """Blocking :meth:`AsyncPromptLibrary.reload`."""
        self._run(self._library.reload())

    def validate(self) -> list[str]:
        """Blocking :meth:`AsyncPromptLibrary.validate`."""
        return self._run(self._library.validate())


def as_async(library: AbstractPromptLibrary) -> AsyncPromptLibrary:
    """Present a synchronous prompt library to an asynchronous consumer.

    The cheap direction: each call is offloaded to a worker thread so the
    consumer's loop is never stalled by a library that reads something to
    answer. Nothing is owned and there is nothing to close.

    Args:
        library: The synchronous library. Its lifetime stays the caller's.

    Returns:
        An :class:`AsyncPromptLibrary` over the same content.
    """
    return AsyncPromptLibraryView(library)


def as_sync(
    library: AsyncPromptLibrary,
    *,
    timeout: float | None = None,
    bridge: SyncLoopBridge | None = None,
) -> SyncPromptLibraryView:
    """Present an asynchronous prompt library to a synchronous consumer.

    The expensive direction. The result owns a private event loop on a daemon
    thread unless handed a ``bridge``, and every call **blocks the calling
    thread** for the library's whole round trip. Close it when done --- ``with``
    and ``async with`` are the context-manager forms.

    **Not for** :class:`AsyncPromptBuilder`: it calls its library synchronously
    from inside its own ``async def``, so this would block the builder's loop
    on every template fetch. See this module's docstring.

    Args:
        library: The asynchronous library to reach. Not the view's to close.
        timeout: Seconds to allow each call, giving a synchronous caller an
            upper bound on a blocking wait it cannot otherwise cancel.
        bridge: A bridge to share with other wrappers, so several of them cost
            one daemon thread between them. A bridge passed here belongs to the
            caller and outlives the view's ``close()``.

    Returns:
        A :class:`SyncPromptLibraryView` --- an :class:`AbstractPromptLibrary`
        that also carries ``close()`` and both context-manager protocols.
    """
    return SyncPromptLibraryView(library, timeout=timeout, bridge=bridge)
