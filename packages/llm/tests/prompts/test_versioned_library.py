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
from typing import Any

import pytest
from dataknobs_common.testing import assert_no_leaked_bridge_threads, live_dk_daemon_threads

from dataknobs_llm.prompts import VersionedPromptLibrary
from dataknobs_llm.prompts.base import (
    AsyncPromptLibrary,
    BasePromptLibrary,
    PromptTemplateDict,
    as_sync,
)
from dataknobs_llm.prompts.implementations import ConfigPromptLibrary
from dataknobs_llm.prompts.versioning import VersionManager
from dataknobs_llm.prompts.versioning.types import PromptVariant, VersionStatus

_BASE_CONFIG = {"system": {"inherited": {"template": "From the base library"}}}


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


# ===== base_library, and the three answers to "which flavour is this?" =====


class _ConsumerLibrary(BasePromptLibrary):
    """A library written the way the mixin's own docstring invites.

    ``BasePromptLibrary`` stopped declaring the interface, so a consumer
    extending it and forgetting to name a flavour produces exactly this: an
    object with every accessor, answering to neither protocol. It is the shape
    the CHANGELOG contemplates, which is why it stands in here for "something
    the discriminator does not recognise" rather than a bare ``object()``.
    """

    def get_system_prompt(self, name: str, **kwargs: Any) -> PromptTemplateDict | None:
        return {"template": "from the consumer's library"} if name == "known" else None

    def get_user_prompt(self, name: str, **kwargs: Any) -> PromptTemplateDict | None:
        return None

    def list_system_prompts(self) -> list[str]:
        return ["known"]

    def list_user_prompts(self) -> list[str]:
        return []

    def get_message_index(self, name: str, **kwargs: Any) -> Any:
        return None

    def list_message_indexes(self) -> list[str]:
        return []

    def get_rag_config(self, name: str, **kwargs: Any) -> Any:
        return None

    def get_prompt_rag_configs(
        self, prompt_name: str, prompt_type: str = "user", **kwargs: Any
    ) -> list[Any]:
        return []


def test_a_base_library_of_neither_flavour_is_refused_at_construction() -> None:
    """The discriminator is total, or it guesses --- and it guessed ``async``.

    ``as_async`` if it is an ``AbstractPromptLibrary``, otherwise store it as
    though it were an ``AsyncPromptLibrary`` is a two-way branch over a
    three-way question. Anything unrecognised took the second arm, constructed
    cleanly, and raised ``TypeError: object ... can't be used in 'await'
    expression`` from a *fallback* path --- reached only for a name that has no
    version, so possibly long after construction and nowhere near it.
    """
    with pytest.raises(TypeError, match=r"AbstractPromptLibrary.*AsyncPromptLibrary"):
        VersionedPromptLibrary(base_library=_ConsumerLibrary())


@pytest.mark.asyncio
async def test_a_synchronous_base_library_is_reached_without_stalling_the_loop() -> None:
    """The flavour this library is most likely to be handed, on a migration."""
    library = VersionedPromptLibrary(base_library=ConfigPromptLibrary(_BASE_CONFIG))

    assert library.base_library is not None
    # The published attribute is the object that was passed, not the view.
    assert isinstance(library.base_library, ConfigPromptLibrary)

    fallback = await library.get_system_prompt("inherited")
    assert fallback is not None
    assert fallback["template"] == "From the base library"
    assert await library.list_system_prompts() == ["inherited"]


@pytest.mark.asyncio
async def test_an_asynchronous_base_library_is_reached_unwrapped() -> None:
    """Already the right flavour, so no door is needed."""
    base = VersionedPromptLibrary()
    await base.create_version(
        name="inherited", prompt_type="system", template="From the base library", version="1.0.0"
    )

    library = VersionedPromptLibrary(base_library=base)

    fallback = await library.get_system_prompt("inherited")
    assert fallback is not None
    assert fallback["template"] == "From the base library"


@pytest.mark.asyncio
async def test_replacing_the_base_library_replaces_what_lookups_reach() -> None:
    """The published attribute and the private view cannot drift apart.

    ``base_library`` was a plain attribute while the view was computed once in
    ``__init__``, so assigning a new library changed what ``get_metadata``
    reported while every lookup kept reaching the old one.
    """
    library = VersionedPromptLibrary(base_library=ConfigPromptLibrary(_BASE_CONFIG))
    library.base_library = ConfigPromptLibrary(
        {"system": {"inherited": {"template": "From the replacement"}}}
    )

    fallback = await library.get_system_prompt("inherited")
    assert fallback is not None
    assert fallback["template"] == "From the replacement"

    library.base_library = None
    assert await library.get_system_prompt("inherited") is None
    assert library.get_metadata()["has_base_library"] is False


def test_replacing_the_base_library_with_neither_flavour_is_refused() -> None:
    """The setter is the same door as the constructor, or it is a hole in it."""
    library = VersionedPromptLibrary()
    with pytest.raises(TypeError, match=r"AbstractPromptLibrary.*AsyncPromptLibrary"):
        library.base_library = _ConsumerLibrary()


# ===== What the listings actually walk =====


@pytest.mark.asyncio
async def test_a_prompt_whose_last_version_was_deleted_stops_being_listed() -> None:
    """Deleting the last version empties the index entry but keeps the key.

    The listing walked the index's *keys*, so the name survived its own last
    version: ``list_system_prompts`` named a prompt that
    ``get_system_prompt`` then answered ``None`` for.
    """
    library = VersionedPromptLibrary()
    version = await library.create_version(
        name="doomed", prompt_type="system", template="Hello!", version="1.0.0"
    )
    assert await library.list_system_prompts() == ["doomed"]

    assert await library.version_manager.delete_version(version.version_id) is True

    assert await library.get_system_prompt("doomed") is None
    assert await library.list_system_prompts() == []


@pytest.mark.asyncio
async def test_a_prompt_name_containing_a_colon_is_still_listed() -> None:
    """The index key is ``f"{name}:{type}"``, so the name is the *left* part.

    Writing and reading both build the key, so a name carrying a colon round
    trips through ``get_system_prompt`` --- but the listing parsed the key back
    with ``split(":", 1)``, which put the rest of the name in the type slot and
    dropped the prompt from its own listing.
    """
    library = VersionedPromptLibrary()
    await library.create_version(
        name="team:greeting", prompt_type="system", template="Hello!", version="1.0.0"
    )

    assert await library.get_system_prompt("team:greeting") is not None
    assert await library.list_system_prompts() == ["team:greeting"]


@pytest.mark.asyncio
async def test_no_bridge_thread_exists_while_the_library_is_answering() -> None:
    """The property the rename claimed and the leak check cannot see.

    ``assert_no_leaked_bridge_threads`` samples on entry and fails on threads
    still alive at *exit*, so a bridge built and closed inside one call passes
    it --- which is precisely what the old per-call bridge did. Observing from
    inside the manager call the accessor is awaiting catches allocation itself.
    """
    library = VersionedPromptLibrary()
    observed: list[list[str]] = []

    class Witness(VersionManager):
        async def get_version(self, *args: Any, **kwargs: Any) -> Any:
            observed.append([t.name for t in live_dk_daemon_threads()])
            return await super().get_version(*args, **kwargs)

    library.version_manager = Witness(None)
    await library.create_version(
        name="greeting", prompt_type="system", template="Hello!", version="1.0.0"
    )

    assert await library.get_system_prompt("greeting") is not None

    assert observed, "the witness never ran, so it proves nothing"
    assert observed == [[] for _ in observed], (
        f"a bridge thread was alive while the library answered: {observed}"
    )


@pytest.mark.asyncio
async def test_a_template_reflects_a_tag_added_after_it_was_first_read() -> None:
    """The library cached a converted template and never invalidated it.

    That cache was invisible while a manager handed back the object it held:
    the cached dictionary aliased the very list ``tag_version`` appended to, so
    the staleness healed itself for exactly the fields that were mutable. Now
    that a load returns a copy, the alias is gone and the cache is a snapshot
    of whatever the version looked like the first time anybody asked --- while
    the store, correctly, holds the tag.
    """
    library = VersionedPromptLibrary()
    version = await library.create_version(
        name="greeting", prompt_type="system", template="Hello {{name}}!", version="1.0.0"
    )

    first = await library.get_system_prompt("greeting")
    assert first is not None
    assert first["metadata"]["tags"] == []

    await library.tag_version(version.version_id, "production")

    again = await library.get_system_prompt("greeting")
    assert again is not None
    assert again["metadata"]["tags"] == ["production"]
