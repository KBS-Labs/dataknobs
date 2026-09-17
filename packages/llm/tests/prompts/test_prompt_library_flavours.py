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

import threading
from typing import Any

import pytest
from dataknobs_common.testing import assert_twin_types_agree

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
    )


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


def test_converting_both_ways_round_trips_a_read(sync_library: ConfigPromptLibrary) -> None:
    """Each door alone preserves what the library answers; together they compose."""
    with as_sync(as_async(sync_library)) as view:
        assert view.get_system_prompt("greet") == sync_library.get_system_prompt("greet")
        assert view.list_user_prompts() == sync_library.list_user_prompts()
        assert view.get_metadata() == sync_library.get_metadata()
