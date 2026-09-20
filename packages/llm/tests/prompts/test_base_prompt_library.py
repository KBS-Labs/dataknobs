# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""``BasePromptLibrary`` is a mixin, and it stops answering for the interface.

Two properties, one change. The mixin re-declared nine of the ABC's abstract
methods and gave eight of them ``NotImplementedError`` bodies --- which is a
*satisfied* abstract method as far as :class:`abc.ABC` is concerned, so a
subclass missing a method constructed fine and failed at the call instead of at
construction. The ninth, ``get_metadata``, is a real implementation over the
mixin's own metadata dict and stays.

And with the eight gone the mixin has nothing of the sync interface left in it,
so it stops declaring that interface: caching, parsing and metadata are useful
to a library of *either* flavour, and a mixin that is itself an
:class:`AbstractPromptLibrary` cannot be reused by an
:class:`AsyncPromptLibrary` without making that class both. The shape is the one
this package already runs one layer down --- ``ResourceAdapterBase`` +
``ResourceAdapter(ResourceAdapterBase, ABC)`` +
``AsyncResourceAdapter(ResourceAdapterBase, ABC)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataknobs_llm.prompts.base import (
    AbstractPromptLibrary,
    AsyncPromptLibrary,
    BasePromptLibrary,
    MessageIndex,
    PromptTemplateDict,
    RAGConfig,
)
from dataknobs_llm.prompts.implementations import (
    ConfigPromptLibrary,
    FileSystemPromptLibrary,
)


class _MissingOneMethod(BasePromptLibrary, AbstractPromptLibrary):
    """Every member of the interface but ``get_user_prompt``.

    Written the way a consumer writes one --- both bases named, because the
    mixin no longer names the interface for them.
    """

    def get_system_prompt(self, name: str, **kwargs: Any) -> PromptTemplateDict | None:
        return None

    def list_system_prompts(self) -> list[str]:
        return []

    def list_user_prompts(self) -> list[str]:
        return []

    def get_message_index(self, name: str, **kwargs: Any) -> MessageIndex | None:
        return None

    def list_message_indexes(self) -> list[str]:
        return []

    def get_rag_config(self, name: str, **kwargs: Any) -> RAGConfig | None:
        return None

    def get_prompt_rag_configs(
        self, prompt_name: str, prompt_type: str = "user", **kwargs: Any
    ) -> list[RAGConfig]:
        return []


def test_a_library_missing_a_method_is_refused_at_construction() -> None:
    """Red before: the stub satisfied the ABC, so this constructed.

    The failure then arrived from ``get_user_prompt`` at whatever point the
    application first asked for a user prompt --- which is the one place the
    author of the class is no longer looking.
    """
    with pytest.raises(TypeError, match="get_user_prompt"):
        _MissingOneMethod()


def test_the_mixin_does_not_answer_for_the_interface() -> None:
    """The mixin is flavour-neutral, so either flavour can reuse it."""
    assert not issubclass(BasePromptLibrary, AbstractPromptLibrary)
    assert not issubclass(BasePromptLibrary, AsyncPromptLibrary)


def test_the_mixin_keeps_the_one_member_it_actually_implements() -> None:
    """``get_metadata`` is the ninth, and deleting all nine would delete it."""

    class Bare(BasePromptLibrary):
        pass

    assert Bare().get_metadata() == {"class": "Bare", "cache_enabled": True}


@pytest.mark.parametrize("library_type", [FileSystemPromptLibrary, ConfigPromptLibrary])
def test_the_shipped_leaves_are_still_prompt_libraries(library_type: type) -> None:
    """The over-correction guard for the three-line base change.

    Both leaves reached the interface *through* the mixin. They name it
    themselves now, so every ``isinstance(x, AbstractPromptLibrary)`` an
    existing consumer holds still answers the same.
    """
    assert issubclass(library_type, AbstractPromptLibrary)
    assert issubclass(library_type, BasePromptLibrary)


class _AsyncMixinLibrary(BasePromptLibrary, AsyncPromptLibrary):
    """The reuse the demotion was *for*, written out.

    Nothing shipped here is this shape yet --- ``VersionedPromptLibrary`` keeps
    a cache of its own, keyed by version id rather than by name --- so without
    a class like this the stated motivation for making the mixin
    flavour-neutral ships with no adopter and nothing checking it holds.
    """

    async def get_system_prompt(self, name: str, **kwargs: Any) -> PromptTemplateDict | None:
        cached = self._get_cached_system_prompt(name)
        if cached is not None:
            return cached
        template = self._parse_prompt_template("Hello {{name}}!")
        self._cache_system_prompt(name, template)
        return template

    async def list_system_prompts(self) -> list[str]:
        return []

    async def get_user_prompt(self, name: str, **kwargs: Any) -> PromptTemplateDict | None:
        return None

    async def list_user_prompts(self) -> list[str]:
        return []

    async def get_message_index(self, name: str, **kwargs: Any) -> MessageIndex | None:
        return None

    async def list_message_indexes(self) -> list[str]:
        return []

    async def get_rag_config(self, name: str, **kwargs: Any) -> RAGConfig | None:
        return None

    async def get_prompt_rag_configs(
        self, prompt_name: str, prompt_type: str = "user", **kwargs: Any
    ) -> list[RAGConfig]:
        return []

    async def reload(self) -> None:
        """This half's flavour over the mixin's flavour-neutral core."""
        self._reload_caches()


@pytest.mark.asyncio
async def test_an_async_library_reuses_the_mixin_without_inheriting_a_flavour() -> None:
    """Construct it, read through its cache, and reload it --- all awaited.

    ``reload`` is the member that made this fail: the mixin carried a plain
    ``def reload`` clearing the caches, which wins the MRO over
    ``AsyncPromptLibrary``'s ``async def`` default. So the class constructed,
    every accessor awaited correctly, and ``await library.reload()`` raised
    ``TypeError: object NoneType can't be used in 'await' expression`` --- a
    flavour smuggled into a mixin whose whole claim is that it has none.
    """
    library = _AsyncMixinLibrary()

    assert await library.get_system_prompt("greet") is not None
    assert library._system_prompt_cache  # the mixin's cache really is in play

    assert await library.reload() is None
    assert not library._system_prompt_cache, "reload reaches the mixin's shared core"


def test_reloading_a_filesystem_library_re_reads_the_filesystem(tmp_path: Path) -> None:
    """``reload`` emptied these libraries instead of reloading them.

    Both shipped leaves keep their whole content in the mixin's caches --- the
    listings return ``self._system_prompt_cache.keys()`` --- so the inherited
    ``reload``, which only *clears* those caches, discarded the library and
    loaded nothing back. A library reloaded from its source is supposed to be
    at least as full afterwards, not empty.
    """
    system = tmp_path / "system"
    system.mkdir()
    (system / "greet.yaml").write_text("template: Hello {{name}}!\n")

    library = FileSystemPromptLibrary(tmp_path)
    assert library.list_system_prompts() == ["greet"]

    (system / "greet.yaml").write_text("template: Good day {{name}}!\n")
    (system / "signoff.yaml").write_text("template: Bye!\n")
    library.reload()

    assert sorted(library.list_system_prompts()) == ["greet", "signoff"]
    template = library.get_system_prompt("greet")
    assert template is not None
    assert template["template"] == "Good day {{name}}!", "reload re-read the changed file"


def test_reloading_a_config_library_re_reads_its_configuration() -> None:
    """The same defect, the same shape, the other leaf."""
    config = {"system": {"greet": {"template": "Hello {{name}}!"}}}
    library = ConfigPromptLibrary(config)
    assert library.list_system_prompts() == ["greet"]

    library.reload()

    assert library.list_system_prompts() == ["greet"]
    template = library.get_system_prompt("greet")
    assert template is not None
    assert template["template"] == "Hello {{name}}!"
