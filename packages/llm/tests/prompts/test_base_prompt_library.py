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

from typing import Any

import pytest

from dataknobs_llm.prompts.base import (
    AbstractPromptLibrary,
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
