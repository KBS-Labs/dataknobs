# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""An object that answers ``obj['field']`` answers ``in``, ``len`` and iteration.

Python does not derive those from ``__getitem__``. It answers ``in`` and
iteration from the *type*, and when a type supplies neither ``__contains__``
nor ``__iter__`` the interpreter falls back to the **legacy integer-indexing
protocol**: it asks for key ``0``, then ``1``, and so on until ``IndexError``.
On a string-keyed record that first question is ``obj[0]``, which raises
``KeyError: 0`` --- an exception naming a key the caller never wrote, from a
line that only asked whether a field was present.

``KeyError`` does not terminate that protocol the way ``IndexError`` does, so
it propagates. Both classes here were measured before the fix:

=========================  ===================  ==============================
                           ``ResolvedConfig``   ``BotContext``
=========================  ===================  ==============================
``'a' in obj``             ``KeyError: 0``      ``True`` (has ``__contains__``)
``len(obj)``               ``TypeError``        ``TypeError``
``list(obj)``              ``KeyError: 0``      ``KeyError: 0``
``dict(obj)``              ``KeyError: 0``      ``KeyError: 0``
``{**obj}``                ``TypeError``        ``TypeError``
``obj.keys()``             ``AttributeError``   ``AttributeError``
=========================  ===================  ==============================

``BotContext`` shows why a half-surface is not half-safe: it *has*
``__contains__``, so the membership test everyone writes first works, and the
trap stays hidden until something iterates --- ``dict(context)`` to log the
request metadata, ``{**context}`` to merge it.

The two classes are completed differently on purpose, and the difference is
the point rather than an inconsistency:

**``ResolvedConfig`` becomes a real** :class:`~collections.abc.Mapping`. Its
subscript surface *is* the resolved configuration --- ``to_dict()`` already
says so by returning exactly that --- so ``isinstance(config, Mapping)``
being true is correct, and lets a resolved config go wherever a config
mapping is expected. It is a ``Mapping`` and not a ``MutableMapping``
because ``ConfigCachingManager.get_or_create`` hands **the same object** to
every caller (``test_cache_hit_returns_same_instance``), so a write through
one caller would be a write into another caller's configuration.

**``BotContext`` gets the surface without the ABC.** Its subscript surface is
a documented convenience over *one field*, ``request_metadata``, while the
object's identity is ``conversation_id``/``client_id``/``user_id``. Declaring
it a ``Mapping`` would make ``isinstance(context, Mapping)`` true for the
193 sites in this workspace that read a mapping as "a bag of data to walk,
merge or serialize" --- and every one of them would silently drop the three
identity fields. That is the same class of silent wrong answer this file
exists to remove, so the surface is completed and the ABC is not claimed.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from dataknobs_bots.bot.context import BotContext
from dataknobs_bots.registry import ConfigCachingManager, InMemoryBackend, ResolvedConfig

RESOLVED = {"llm": {"provider": "ollama"}, "memory": {"backend": "buffer"}}


def _config() -> ResolvedConfig:
    return ResolvedConfig(
        config_id="test-1",
        raw_config={"llm": {"$resource": "default_llm"}},
        resolved_config=dict(RESOLVED),
        environment_name="production",
    )


def _context() -> BotContext:
    return BotContext(
        conversation_id="conv-1",
        client_id="client-1",
        request_metadata={"trace_id": "t-9", "locale": "en"},
    )


# --------------------------------------------------------------------------- #
# ResolvedConfig
# --------------------------------------------------------------------------- #


def test_a_resolved_config_can_be_asked_whether_a_section_is_present() -> None:
    """``'llm' in config`` reached the integer-index fallback and raised.

    The caller asked about a section by name and got back ``KeyError: 0``,
    naming an integer key that appears nowhere in the configuration.
    """
    config = _config()

    assert "llm" in config
    assert "missing" not in config


def test_a_resolved_config_reports_how_many_sections_it_has() -> None:
    config = _config()

    assert len(config) == 2


def test_a_resolved_config_can_be_iterated() -> None:
    config = _config()

    assert sorted(config) == ["llm", "memory"]
    assert sorted(config.keys()) == ["llm", "memory"]
    assert sorted(dict(config.items())) == ["llm", "memory"]
    assert list(config.values()) == [RESOLVED["llm"], RESOLVED["memory"]]


def test_a_resolved_config_can_be_spread_into_a_dict() -> None:
    """``{**config}`` is how a caller layers a resolved config under overrides."""
    config = _config()

    assert dict(config) == RESOLVED
    assert {**config, "memory": {"backend": "vector"}} == {
        "llm": {"provider": "ollama"},
        "memory": {"backend": "vector"},
    }


def test_a_resolved_config_is_a_mapping() -> None:
    """So it can be handed anywhere a configuration mapping is expected."""
    config = _config()

    assert isinstance(config, Mapping)


def test_a_resolved_config_equals_the_configuration_it_resolved_to() -> None:
    config = _config()

    assert config == RESOLVED


def test_an_empty_resolved_config_is_falsy() -> None:
    """``bool`` follows ``__len__``; with neither, an empty config read as true."""
    assert not ResolvedConfig(config_id="empty", raw_config={}, resolved_config={})
    assert _config()


def test_a_resolved_config_shows_its_sections_when_printed() -> None:
    """A bare ``object`` repr in a log says only that something went wrong."""
    printed = repr(_config())

    assert "test-1" in printed
    assert "production" in printed
    assert "llm" in printed


def test_a_resolved_config_refuses_writes() -> None:
    """The cache hands one instance to every caller, so a write is another's read.

    Deliberately a ``Mapping`` rather than a ``MutableMapping``: see
    ``test_a_shared_cached_config_cannot_be_written_through``, which drives the
    same guarantee through the manager that does the sharing.
    """
    config = _config()

    with pytest.raises(TypeError):
        config["llm"] = {"provider": "openai"}  # type: ignore[index]


async def test_a_shared_cached_config_cannot_be_written_through() -> None:
    """Through the real manager: two callers, one object, no write path."""
    backend = InMemoryBackend()
    await backend.register("shared-bot", dict(RESOLVED))
    manager = ConfigCachingManager(backend=backend)
    await manager.initialize()
    try:
        first = await manager.get_or_create("shared-bot")
        second = await manager.get_or_create("shared-bot")

        assert first is second, "the manager caches one instance per config id"
        assert dict(first) == RESOLVED
        with pytest.raises(TypeError):
            second["llm"] = {"provider": "openai"}  # type: ignore[index]

        assert dict(first) == RESOLVED
    finally:
        await manager.close()


def test_a_resolved_config_still_deep_copies_on_to_dict() -> None:
    """``dict(config)`` is shallow; ``to_dict()`` is the copy, and stays one."""
    config = _config()

    detached = config.to_dict()
    detached["llm"]["provider"] = "openai"

    assert config["llm"]["provider"] == "ollama"


# --------------------------------------------------------------------------- #
# BotContext
# --------------------------------------------------------------------------- #


def test_a_context_can_be_iterated() -> None:
    """``for key in context`` asked ``request_metadata[0]`` and raised."""
    context = _context()

    assert sorted(context) == ["locale", "trace_id"]
    assert sorted(context.keys()) == ["locale", "trace_id"]


def test_a_context_reports_how_much_request_metadata_it_carries() -> None:
    context = _context()

    assert len(context) == 2
    assert len(BotContext(conversation_id="c", client_id="cl")) == 0


def test_a_context_can_be_turned_into_its_request_metadata() -> None:
    """``dict(context)`` to log it, ``{**context}`` to merge it."""
    context = _context()

    assert dict(context) == {"trace_id": "t-9", "locale": "en"}
    assert {**context, "locale": "fr"} == {"trace_id": "t-9", "locale": "fr"}
    assert list(context.values()) == ["t-9", "en"]
    assert dict(context.items()) == {"trace_id": "t-9", "locale": "en"}


def test_a_context_is_not_a_mapping() -> None:
    """Deliberate: ``dict(context)`` is one field, not the context.

    Anything that reads "a mapping" as the whole object --- serializing it,
    merging it, walking it --- would drop ``conversation_id``, ``client_id``
    and ``user_id`` without saying so.
    """
    context = _context()

    assert not isinstance(context, Mapping)
    assert context.conversation_id == "conv-1"
    assert context.client_id == "client-1"


def test_a_context_is_always_truthy() -> None:
    """``__len__`` would otherwise have made an empty-metadata context falsy.

    A context is not its request metadata: it always carries a conversation
    and a client, so there is no state in which it is "empty". Every
    ``if context:`` guard written before this file keeps meaning what it
    meant; ``len(context)`` is how emptiness is asked about now.
    """
    empty = BotContext(conversation_id="conv-1", client_id="client-1")

    assert empty
    assert len(empty) == 0
    assert _context()


def test_a_context_copy_still_copies_the_context() -> None:
    """``copy`` predates the mapping surface and keeps its own meaning.

    It is not a mapping's ``copy`` and returns no dict: it clones the whole
    context with field overrides, which is why ``BotContext`` does not claim
    the ABC that would make a reader expect otherwise.
    """
    context = _context()

    clone = context.copy(conversation_id="conv-2")

    assert isinstance(clone, BotContext)
    assert clone.conversation_id == "conv-2"
    assert clone.client_id == "client-1"
    assert dict(clone) == dict(context)


def test_the_documented_context_access_still_works() -> None:
    """The surface that already worked, guarded against the completion."""
    context = _context()

    assert context["trace_id"] == "t-9"
    assert context.get("trace_id") == "t-9"
    assert context.get("absent", "fallback") == "fallback"
    assert "trace_id" in context
    assert "absent" not in context

    context["attempt"] = 2

    assert context.request_metadata["attempt"] == 2
    assert len(context) == 3
