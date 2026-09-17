# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for :class:`VersionedPromptLibrary`.

The class shipped for some time with no test touching it at all, and it was
unconstructible the whole while: ``AbstractPromptLibrary.reload`` carried an
``@abstractmethod`` that its own docstring contradicted, so every one of this
library's twenty-odd methods was unreachable. Construction is therefore the
first thing asserted here, and the rest of the file is the happy path that
would have caught it.

It is an :class:`AsyncPromptLibrary` now. The accessors were synchronous
because the only protocol available said so, and they reached an asynchronous
version manager --- first with ``loop.run_until_complete``, which raised on any
caller already on a loop, then through a per-call bridge, which merely blocked
that caller's loop instead. Neither is a property of the library; both are
what a synchronous signature over a store costs. A ``def`` caller now goes
through :func:`~dataknobs_llm.prompts.base.views.as_sync` and pays the bridge
*visibly*, which is the point --- see ``test_prompt_library_flavours.py`` for
the doors themselves.
"""

import asyncio
import inspect
import threading

import pytest
from dataknobs_common.testing import assert_no_leaked_bridge_threads

from dataknobs_llm.prompts import VersionedPromptLibrary
from dataknobs_llm.prompts.base import AsyncPromptLibrary, as_sync
from dataknobs_llm.prompts.versioning.types import PromptVariant, VersionStatus


def test_the_library_can_be_constructed() -> None:
    """The shipped implementation is concrete, not abstract."""
    assert not inspect.isabstract(VersionedPromptLibrary), (
        f"unimplemented: {sorted(VersionedPromptLibrary.__abstractmethods__)}"
    )
    library = VersionedPromptLibrary()
    assert isinstance(library, AsyncPromptLibrary)


@pytest.mark.asyncio
async def test_reload_is_an_optional_hook_with_a_no_op_default() -> None:
    """The base declares reload optional; the default must therefore exist."""
    assert not getattr(AsyncPromptLibrary.reload, "__isabstractmethod__", False), (
        "reload is documented as optional with a do-nothing default"
    )
    assert await VersionedPromptLibrary().reload() is None


@pytest.mark.asyncio
async def test_a_version_round_trips_through_the_library() -> None:
    library = VersionedPromptLibrary()

    created = await library.create_version(
        name="greeting",
        prompt_type="system",
        template="Hello {{name}}!",
        version="1.0.0",
    )
    assert created.version == "1.0.0"
    assert created.version_id, "create_version mints the id the dataclass requires"

    fetched = await library.get_version("greeting", "system", "1.0.0")
    assert fetched is not None
    assert fetched.template == "Hello {{name}}!"

    versions = await library.list_versions("greeting", "system")
    assert [v.version for v in versions] == ["1.0.0"]


@pytest.mark.asyncio
async def test_the_accessors_are_awaited_from_the_code_that_populates_the_library() -> None:
    """One flavour, end to end.

    Every writer on this library is a coroutine, so a running loop is the only
    place it can be populated from --- which used to be exactly the place its
    readers could not be called. Reader and writer are now the same flavour and
    the sequence in the class's own usage example just works.
    """
    library = VersionedPromptLibrary()
    await library.create_version(
        name="greeting",
        prompt_type="system",
        template="Hello {{name}}!",
        version="1.0.0",
    )
    await library.create_version(
        name="greeting",
        prompt_type="user",
        template="Hi {{name}}!",
        version="1.0.0",
    )

    system = await library.get_system_prompt("greeting", version="1.0.0")
    user = await library.get_user_prompt("greeting", version="1.0.0")

    assert system is not None
    assert system["template"] == "Hello {{name}}!"
    assert user is not None
    assert user["template"] == "Hi {{name}}!"


@pytest.mark.asyncio
async def test_reading_the_library_allocates_no_bridge_thread() -> None:
    """The cost the flavour removes rather than relocates.

    A synchronous accessor over an async manager had to reach a loop somehow,
    and with no ``close()`` on the protocol to hang a held bridge from, that
    meant a throwaway daemon thread **per call**. An awaited accessor runs on
    the caller's own loop and allocates nothing.
    """
    library = VersionedPromptLibrary()
    await library.create_version(
        name="greeting",
        prompt_type="system",
        template="Hello {{name}}!",
        version="1.0.0",
    )

    with assert_no_leaked_bridge_threads():
        assert await library.get_system_prompt("greeting", version="1.0.0") is not None
        assert await library.get_system_prompt("missing", version="1.0.0") is None


def test_a_synchronous_caller_reaches_the_library_through_the_named_door() -> None:
    """``def`` code still has a route, and the bridge it costs is visible."""
    library = VersionedPromptLibrary()
    asyncio.run(
        library.create_version(
            name="greeting",
            prompt_type="system",
            template="Hello {{name}}!",
            version="1.0.0",
        )
    )

    with as_sync(library) as view:
        template = view.get_system_prompt("greeting", version="1.0.0")

    assert template is not None
    # PromptTemplateDict is a TypedDict, so this is subscript access, not attribute
    assert template["template"] == "Hello {{name}}!"


@pytest.mark.asyncio
async def test_a_user_gets_one_variant_and_keeps_it() -> None:
    library = VersionedPromptLibrary()
    for version, template in (("1.0.0", "Hello {{name}}!"), ("1.0.1", "Hi {{name}}!")):
        await library.create_version(
            name="greeting",
            prompt_type="system",
            template=template,
            version=version,
            status=VersionStatus.ACTIVE,
        )

    experiment = await library.create_experiment(
        name="greeting",
        prompt_type="system",
        variants=[
            PromptVariant("1.0.0", 0.5, "Control"),
            PromptVariant("1.0.1", 0.5, "Treatment"),
        ],
    )
    assert experiment.traffic_split == {"1.0.0": 0.5, "1.0.1": 0.5}

    assigned = await library.get_variant_for_user(experiment.experiment_id, "user123")
    assert assigned in {"1.0.0", "1.0.1"}
    # Sticky: the same user is not reassigned on a later turn.
    again = await library.get_variant_for_user(experiment.experiment_id, "user123")
    assert again == assigned

    chosen = await library.get_version("greeting", "system", assigned)
    assert chosen is not None


@pytest.mark.asyncio
async def test_the_two_listings_report_the_same_shape() -> None:
    """Both walk one index and differ only in the type they filter on."""
    library = VersionedPromptLibrary()
    await library.create_version(
        name="greeting", prompt_type="system", template="Hello!", version="1.0.0"
    )
    await library.create_version(
        name="signoff", prompt_type="user", template="Bye!", version="1.0.0"
    )

    assert await library.list_system_prompts() == ["greeting"]
    assert await library.list_user_prompts() == ["signoff"]


def test_the_sync_door_installs_no_event_loop_in_the_calling_thread() -> None:
    """A read must not leave a loop behind in whatever thread asked for it.

    ``list_system_prompts`` once opened with a ``get_event_loop``/
    ``new_event_loop`` preamble whose ``loop`` was then never used -- so on any
    thread without one it constructed a loop, installed it thread-globally with
    ``set_event_loop``, ran nothing on it and never closed it. The listing is a
    coroutine now and has no such preamble to drift back into, but the property
    belongs to whatever a ``def`` caller reaches, so it moves to the door.
    """
    library = VersionedPromptLibrary()
    observed: dict[str, object] = {}

    def call_on_a_bare_thread() -> None:
        # A non-main thread starts with no current loop, so the policy raises
        # rather than fabricating one -- which makes "was a loop installed?"
        # answerable here and nowhere else.
        try:
            observed["before"] = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError as exc:
            observed["before"] = exc

        with as_sync(library) as view:
            observed["result"] = view.list_system_prompts()

        try:
            observed["after"] = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError as exc:
            observed["after"] = exc

    thread = threading.Thread(target=call_on_a_bare_thread)
    thread.start()
    thread.join()

    assert isinstance(observed["before"], RuntimeError), "thread should start loopless"
    assert observed["result"] == []
    assert isinstance(observed["after"], RuntimeError), (
        f"a loop was installed and left running: {observed['after']!r}"
    )
