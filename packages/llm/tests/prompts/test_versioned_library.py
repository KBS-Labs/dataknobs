# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for :class:`VersionedPromptLibrary`.

The class shipped for some time with no test touching it at all, and it was
unconstructible the whole while: ``AbstractPromptLibrary.reload`` carried an
``@abstractmethod`` that its own docstring contradicted, so every one of this
library's twenty-odd methods was unreachable. Construction is therefore the
first thing asserted here, and the rest of the file is the happy path that
would have caught it.
"""

import asyncio
import inspect

import pytest

from dataknobs_llm.prompts import VersionedPromptLibrary
from dataknobs_llm.prompts.base import AbstractPromptLibrary
from dataknobs_llm.prompts.versioning.types import PromptVariant, VersionStatus


def test_the_library_can_be_constructed() -> None:
    """The shipped implementation is concrete, not abstract."""
    assert not inspect.isabstract(VersionedPromptLibrary), (
        f"unimplemented: {sorted(VersionedPromptLibrary.__abstractmethods__)}"
    )
    library = VersionedPromptLibrary()
    assert isinstance(library, AbstractPromptLibrary)


def test_reload_is_an_optional_hook_with_a_no_op_default() -> None:
    """The base declares reload optional; the default must therefore exist."""
    assert not getattr(AbstractPromptLibrary.reload, "__isabstractmethod__", False), (
        "reload is documented as optional with a do-nothing default"
    )
    assert VersionedPromptLibrary().reload() is None


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
async def test_the_sync_accessors_cannot_be_called_from_async_code() -> None:
    """A second, separate defect -- pinned here rather than left to surprise.

    ``get_system_prompt`` and ``get_user_prompt`` are synchronous, to satisfy
    :class:`AbstractPromptLibrary`, and reach their async version manager with
    ``loop.run_until_complete``. On a running loop that raises, and a running
    loop is the only place this library can be populated from, since every
    writer on it is a coroutine. So the two inherited accessors are
    unreachable in practice.

    This test documents the behaviour as it stands. It is not an endorsement:
    when the sync/async bridge is redesigned, invert it.
    """
    library = VersionedPromptLibrary()
    await library.create_version(
        name="greeting",
        prompt_type="system",
        template="Hello {{name}}!",
        version="1.0.0",
    )

    with pytest.raises(RuntimeError, match="already running"):
        library.get_system_prompt("greeting", version="1.0.0")

    with pytest.raises(RuntimeError, match="already running"):
        library.get_user_prompt("greeting", version="1.0.0")


def test_the_sync_accessors_do_work_off_a_running_loop() -> None:
    """Off a loop they behave -- which is why the defect above is easy to miss."""
    library = VersionedPromptLibrary()
    asyncio.run(
        library.create_version(
            name="greeting",
            prompt_type="system",
            template="Hello {{name}}!",
            version="1.0.0",
        )
    )

    template = library.get_system_prompt("greeting", version="1.0.0")
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
