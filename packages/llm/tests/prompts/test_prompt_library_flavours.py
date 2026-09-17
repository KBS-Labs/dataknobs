# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Two prompt-library flavours, one surface, and the two doors between them.

``AbstractPromptLibrary`` declares every accessor ``def``. That is right for a
library whose content is in memory by the time it exists and wrong for one that
fronts a store, which has to await to answer --- and a synchronous signature
left such a library only bad choices: block the caller's thread on a private
loop, or refuse outright when the caller was already on one.

So the flavour is declared. What is checked here is that declaring it did not
produce two surfaces: the guard compares the halves member by member with the
one deliberate difference stated rather than discovered, and the doors are
checked in both directions, including from inside a running loop --- the place
the synchronous accessors used to raise.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
from dataknobs_common.sync_bridge import SyncLoopBridge
from dataknobs_common.testing import assert_no_leaked_bridge_threads, assert_twin_types_agree

from dataknobs_llm.prompts import VersionedPromptLibrary
from dataknobs_llm.prompts.base import (
    AbstractPromptLibrary,
    AsyncPromptLibrary,
    as_async,
    as_sync,
)
from dataknobs_llm.prompts.implementations import ConfigPromptLibrary

CONFIG = {
    "system": {"greet": {"template": "Hello {{name}}!"}},
    "user": {"ask": {"template": "Tell me about {{topic}}"}},
}


@pytest.fixture
def sync_library() -> ConfigPromptLibrary:
    """A real synchronous library --- no mock, no fake."""
    return ConfigPromptLibrary(CONFIG)


@pytest.fixture
async def async_library() -> VersionedPromptLibrary:
    """A real asynchronous library, populated the only way it can be."""
    library = VersionedPromptLibrary()
    await library.create_version(
        name="greet", prompt_type="system", template="Hello {{name}}!", version="1.0.0"
    )
    await library.create_version(
        name="ask", prompt_type="user", template="Tell me about {{topic}}", version="1.0.0"
    )
    return library


def test_the_two_protocols_declare_one_surface() -> None:
    """Eleven members, one stated difference.

    ``get_metadata`` answers from the library's own configuration rather than
    from its content, so it reaches nothing and is synchronous on both halves.
    Declaring that here is what makes a *second* divergence fail rather than
    quietly join the first.
    """
    assert_twin_types_agree(
        AbstractPromptLibrary,
        AsyncPromptLibrary,
        members=[
            "get_system_prompt",
            "list_system_prompts",
            "get_user_prompt",
            "list_user_prompts",
            "get_message_index",
            "list_message_indexes",
            "get_rag_config",
            "get_prompt_rag_configs",
            "get_metadata",
            "reload",
            "validate",
        ],
        unflavoured_members=["get_metadata"],
        compare_return=True,
    )


def test_the_versioned_library_is_the_async_flavour() -> None:
    """It fronts a store, so it is the flavour that can await one.

    Before this it satisfied the synchronous protocol and reached an
    asynchronous version manager through a per-call bridge --- reachable, but
    blocking whichever loop the caller was on for the whole lookup.
    """
    library = VersionedPromptLibrary()
    assert isinstance(library, AsyncPromptLibrary)
    assert not isinstance(library, AbstractPromptLibrary)


@pytest.mark.asyncio
async def test_a_sync_library_answers_through_the_async_door(
    sync_library: ConfigPromptLibrary,
) -> None:
    library = as_async(sync_library)

    system = await library.get_system_prompt("greet")
    user = await library.get_user_prompt("ask")

    assert system is not None
    assert system["template"] == "Hello {{name}}!"
    assert user is not None
    assert user["template"] == "Tell me about {{topic}}"
    assert await library.list_system_prompts() == ["greet"]
    assert await library.list_user_prompts() == ["ask"]
    assert await library.get_message_index("nothing") is None
    assert await library.list_message_indexes() == []
    assert await library.get_rag_config("nothing") is None
    assert await library.get_prompt_rag_configs("greet", "system") == []
    assert await library.validate() == []
    assert await library.reload() is None
    assert library.get_metadata()["class"] == "ConfigPromptLibrary"


@pytest.mark.asyncio
async def test_the_async_door_does_not_run_the_read_on_the_loop_thread(
    sync_library: ConfigPromptLibrary,
) -> None:
    """A synchronous library may read a file to answer; the loop must not wait.

    The three libraries shipped here answer from memory, so the offload buys
    nothing *for them*. It is not written for them: ``as_async`` is handed any
    :class:`AbstractPromptLibrary`, a shipped ``async def`` cannot see who else
    is on its loop, and a co-tenant stalled by a disk read is the harm
    ``.claude/rules/async-transport.md`` exists to prevent.
    """
    seen: list[str] = []

    class Witness(ConfigPromptLibrary):
        def get_system_prompt(self, name: str, **kwargs: Any) -> Any:
            seen.append(threading.current_thread().name)
            return super().get_system_prompt(name, **kwargs)

    await as_async(Witness(CONFIG)).get_system_prompt("greet")

    assert seen and threading.current_thread().name not in seen


def test_an_async_library_answers_through_the_sync_door_off_a_loop() -> None:
    """The ``def`` caller this door exists for."""
    library = VersionedPromptLibrary()
    asyncio.run(
        library.create_version(
            name="greet", prompt_type="system", template="Hello {{name}}!", version="1.0.0"
        )
    )

    with as_sync(library) as view:
        template = view.get_system_prompt("greet")
        assert template is not None
        assert template["template"] == "Hello {{name}}!"
        assert view.list_system_prompts() == ["greet"]
        assert view.get_metadata()["type"] == "VersionedPromptLibrary"


@pytest.mark.asyncio
async def test_the_sync_door_works_from_inside_a_running_loop(
    async_library: VersionedPromptLibrary,
) -> None:
    """The property the accessors used to fail outright.

    A synchronous call reaching an async library used to raise
    ``RuntimeError: This event loop is already running`` --- and a running loop
    is the only place such a library can be populated from, since every writer
    on it is a coroutine. The bridge runs the coroutine on its own loop
    instead, so the call returns rather than raising.

    It still **blocks** this task's loop for the duration. That is what the
    door costs and why the async library is the better answer where a consumer
    can take one.
    """
    with as_sync(async_library) as view:
        template = view.get_system_prompt("greet")

    assert template is not None
    assert template["template"] == "Hello {{name}}!"


@pytest.mark.asyncio
async def test_the_sync_door_leaves_no_thread_behind(
    async_library: VersionedPromptLibrary,
) -> None:
    """A bridge is a daemon thread with a lifetime, and ``close`` ends it."""
    with assert_no_leaked_bridge_threads():
        with as_sync(async_library) as view:
            assert view.get_system_prompt("greet") is not None


@pytest.mark.asyncio
async def test_several_doors_can_share_one_thread(
    async_library: VersionedPromptLibrary,
) -> None:
    """A bridge passed in belongs to the caller and outlives the view."""
    with assert_no_leaked_bridge_threads(), SyncLoopBridge() as bridge:
        first = as_sync(async_library, bridge=bridge)
        second = as_sync(async_library, bridge=bridge)
        try:
            assert first.get_system_prompt("greet") is not None
            assert second.get_user_prompt("ask") is not None
        finally:
            first.close()
            second.close()
        # Closing both views left the caller's bridge running.
        assert bridge.run(_one()) == 1


@pytest.mark.asyncio
async def test_a_timeout_bounds_a_wait_the_caller_cannot_cancel() -> None:
    """A blocked ``def`` caller has no cancellation of its own."""

    class Stalls(VersionedPromptLibrary):
        async def get_system_prompt(self, name: str, **kwargs: Any) -> Any:
            await asyncio.sleep(30)
            return None  # pragma: no cover - never reached

    with as_sync(Stalls(), timeout=0.1) as view, pytest.raises(TimeoutError):
        view.get_system_prompt("greet")


def test_converting_both_ways_round_trips_a_read(sync_library: ConfigPromptLibrary) -> None:
    """Each door alone preserves what the library answers; together they compose."""
    with as_sync(as_async(sync_library)) as view:
        assert view.get_system_prompt("greet") == sync_library.get_system_prompt("greet")
        assert view.list_user_prompts() == sync_library.list_user_prompts()
        assert view.get_metadata() == sync_library.get_metadata()


async def _one() -> int:
    """A coroutine whose only job is to prove a bridge is still running."""
    return 1


# ===== Positive controls for the guard above =====
#
# The parity assertion is the only thing standing between "one surface" and a
# second silent divergence, and an assertion that has never been observed to
# fail is not evidence that it can. These three drive it against pairs that are
# wrong in each of the ways the real pair could go wrong.


class _SyncHalf:
    """The synchronous half of a deliberately broken twin."""

    def fetch(self, name: str) -> str:
        return name

    def label(self) -> str:
        return "sync"


class _DriftedHalf:
    """Its twin, carrying an undeclared extra parameter."""

    async def fetch(self, name: str, retries: int = 0) -> str:
        return name

    async def label(self) -> str:
        return "async"


class _UnflavouredHalf:
    """Its twin, where ``label`` really *is* flavoured."""

    async def fetch(self, name: str) -> str:
        return name

    async def label(self) -> str:
        return "async"


class _RetypedHalf:
    """Its twin, agreeing on every parameter and not on what it returns."""

    async def fetch(self, name: str) -> bytes:
        return name.encode()

    def label(self) -> str:
        return "async"


def test_the_guard_fires_on_a_drifted_signature() -> None:
    """A parameter on one half only is what the guard is mostly for."""
    with pytest.raises(AssertionError, match="retries"):
        assert_twin_types_agree(_SyncHalf, _DriftedHalf, members=["fetch"])


def test_the_guard_fires_on_a_member_wrongly_declared_unflavoured() -> None:
    """The declaration is compared against what is observed, both ways.

    Declaring ``get_metadata`` unflavoured is what lets the real pair pass, so
    the cost of getting that declaration wrong has to be a failure and not a
    quiet exemption --- otherwise the list is a way to switch the guard off one
    member at a time.
    """
    with pytest.raises(AssertionError, match="declared unflavoured"):
        assert_twin_types_agree(
            _SyncHalf,
            _UnflavouredHalf,
            members=["label"],
            unflavoured_members=["label"],
        )


def test_the_guard_fires_on_a_drifted_return_annotation() -> None:
    """``compare_return`` is on for the real pair, so it must have teeth."""
    with pytest.raises(AssertionError, match=r"str.*bytes"):
        assert_twin_types_agree(_SyncHalf, _RetypedHalf, members=["fetch"], compare_return=True)
