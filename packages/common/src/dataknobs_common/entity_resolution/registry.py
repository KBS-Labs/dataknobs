# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Where a rung is looked up by the name a consumer writes as ``kind:``.

**Two registries, one per flavour, each guarded.** Not one registry holding
both twins: the guard that compares a registration against a protocol skips
``name`` (a property carries no async-ness to disagree about), agrees on
``narrows()``, and separates the pair on the two ``candidates`` members -- so
a single ``validate_type`` could not hold both anyway.

**Where a wrong-flavour registration fails depends on how it was registered,
and both are loud.** A *class* factory is judged when it is registered, and
the refusal names the member and the direction. A *callable* factory cannot
be judged then -- nothing knows what it will return -- so it is judged on the
instance at ``create()``, with the same message. Neither reaches the caller's
``await``, which is the property that matters; the distinction is recorded
because "fails at registration" is true of only one of the two paths, and the
rungs below are registered by the other.

The registries live in ``common``, beside the protocols, and that is forced
rather than chosen: a rung reading a vector store belongs to one package and a
rung reading a lexicon to another, those two are siblings, and only a registry
beside the protocol is reachable from both.
"""

from __future__ import annotations

from typing import Any

from dataknobs_common.entity_resolution.protocols import AsyncMatchSignal, MatchSignal
from dataknobs_common.entity_resolution.signals import (
    AliasSignal,
    AsyncAliasSignal,
    AsyncExactNormalizedSignal,
    AsyncLexicalSignal,
    AsyncScanningSignal,
    ExactNormalizedSignal,
    LexicalSignal,
    ScanningSignal,
)
from dataknobs_common.registry import PluginRegistry

__all__ = ["async_signal_backends", "signal_backends"]


#: Rungs a synchronous cascade can be built from.
signal_backends: PluginRegistry[MatchSignal] = PluginRegistry(
    name="signal_backends",
    validate_type=MatchSignal,
    config_key="kind",
    not_found_kind="rung kind",
)

#: Rungs an asynchronous cascade can be built from.
async_signal_backends: PluginRegistry[AsyncMatchSignal] = PluginRegistry(
    name="async_signal_backends",
    validate_type=AsyncMatchSignal,
    config_key="kind",
    not_found_kind="rung kind",
)


def _make_exact(config: dict[str, Any]) -> MatchSignal:
    return ExactNormalizedSignal(config["entities"], normalizer=config.get("normalizer"))


def _make_alias(config: dict[str, Any]) -> MatchSignal:
    return AliasSignal(config["entities"], normalizer=config.get("normalizer"))


def _make_scan(config: dict[str, Any]) -> MatchSignal:
    # ``max_window`` is forwarded because a document is where a vocabulary the
    # source cannot bound gets its cap -- a caller reaching this factory by
    # writing ``kind: scan`` has no other way to supply one.
    return ScanningSignal(
        config["entities"],
        normalizer=config.get("normalizer"),
        max_window=config.get("max_window"),
    )


def _make_lexical(config: dict[str, Any]) -> MatchSignal:
    # ``threshold``, ``scorer`` and ``max_query_tokens`` are forwarded for
    # ``max_window``'s reason on the rung above: a caller reaching this
    # factory by writing ``kind: lexical`` has no other way to supply any of
    # them, and the scorer in particular is the whole of this rung's
    # published answer to a consumer whose vocabulary is too large for the
    # standard library's.
    #
    # Forwarded, **not** resolved. ``scorer`` and ``normalizer`` are
    # callables and a document cannot write one, so the value arriving here
    # from a parsed document is a string however it was spelled -- which the
    # rung refuses by name rather than letting it reach a call site deep in
    # the scan. Resolving a dotted path into the function it names would let
    # a document reach any importable callable, which is a wider decision
    # than this one.
    return LexicalSignal(
        config["entities"],
        threshold=config.get("threshold", 0.85),
        scorer=config.get("scorer"),
        normalizer=config.get("normalizer"),
        max_query_tokens=config.get("max_query_tokens"),
    )


def _make_async_exact(config: dict[str, Any]) -> AsyncMatchSignal:
    return AsyncExactNormalizedSignal(config["entities"], normalizer=config.get("normalizer"))


def _make_async_alias(config: dict[str, Any]) -> AsyncMatchSignal:
    return AsyncAliasSignal(config["entities"], normalizer=config.get("normalizer"))


def _make_async_scan(config: dict[str, Any]) -> AsyncMatchSignal:
    return AsyncScanningSignal(
        config["entities"],
        normalizer=config.get("normalizer"),
        max_window=config.get("max_window"),
    )


def _make_async_lexical(config: dict[str, Any]) -> AsyncMatchSignal:
    return AsyncLexicalSignal(
        config["entities"],
        threshold=config.get("threshold", 0.85),
        scorer=config.get("scorer"),
        normalizer=config.get("normalizer"),
        max_query_tokens=config.get("max_query_tokens"),
    )


#: Every rung declares its flavour and whether it needs I/O, so a door can
#: refuse a composition *before* building anything. A capability read off an
#: instance is read too late to refuse with.
_DECLARED_METADATA = {"flavour": "sync", "needs_io": False}
_ASYNC_DECLARED_METADATA = {"flavour": "async", "needs_io": False}


def _declared(rung: type[Any], base: dict[str, Any]) -> dict[str, Any]:
    """This rung's registered metadata, with the facts read off the class.

    ``reads_surface_forms`` is a property of the rung and is enforced by the
    rung -- it refuses at construction over a source that withholds the
    capability. It is *also* the fact a caller holding a composition and no
    instances needs, which is the only reason it appears here as well: a
    loader binding a live source refuses the document before anything is
    built, and a registry is the one place it can ask.

    ``bounded_by_longest_form`` is here for the same reason and answers a
    different door's question: whether this rung's cost depends on a number
    the source may not be able to supply. Both are derived rather than
    restated, because two spellings of one fact drift and the drift is
    silent: a rung marked here and not on the class refuses nothing, and a
    rung marked on the class and not here is not refused early.
    """
    return dict(
        base,
        reads_surface_forms=rung.reads_surface_forms,
        bounded_by_longest_form=rung.bounded_by_longest_form,
    )


signal_backends.register(
    "exact", _make_exact, metadata=_declared(ExactNormalizedSignal, _DECLARED_METADATA)
)
signal_backends.register("alias", _make_alias, metadata=_declared(AliasSignal, _DECLARED_METADATA))
signal_backends.register("scan", _make_scan, metadata=_declared(ScanningSignal, _DECLARED_METADATA))
signal_backends.register(
    "lexical", _make_lexical, metadata=_declared(LexicalSignal, _DECLARED_METADATA)
)
async_signal_backends.register(
    "exact",
    _make_async_exact,
    metadata=_declared(AsyncExactNormalizedSignal, _ASYNC_DECLARED_METADATA),
)
async_signal_backends.register(
    "alias", _make_async_alias, metadata=_declared(AsyncAliasSignal, _ASYNC_DECLARED_METADATA)
)
async_signal_backends.register(
    "scan", _make_async_scan, metadata=_declared(AsyncScanningSignal, _ASYNC_DECLARED_METADATA)
)
async_signal_backends.register(
    "lexical",
    _make_async_lexical,
    metadata=_declared(AsyncLexicalSignal, _ASYNC_DECLARED_METADATA),
)


# A rung that exists in one flavour only is *declared* in the other with a
# reason, so asking the synchronous registry for it says why rather than
# "unknown key" -- which would send the reader looking for a typo in a name
# spelled correctly.
#
# Declared here, at this module's import, rather than by the package that
# implements the rung: that package is not a dependency of this one and a test
# in this package need not have imported it, so a mark left to it would be
# present in an application and absent in the leg's own test. Importing a key
# and a sentence costs nothing, which is why the fact can live here even
# though the class cannot.
#
# The reason is the **fact** and says nothing about doors. A door composes the
# sentence naming itself and its twin; this registry is also read by callers
# that have no door at all.
#
# ``lexical`` carries no mark, and the absence is a ruling rather than an
# oversight: the rung lives in this distribution, in the module this registry
# imports, and exists in both flavours -- so neither condition below applies
# and the key is simply registered. It is worth saying because the rung is the
# one here that can need an install, and the install it can need buys *speed*
# rather than the rung: ``scorer`` takes a faster implementation and the
# standard library answers the same way without one.
#
# Two members, and they are marked for **different** conditions, which is why
# the reason is a sentence rather than a flag. ``semantic`` is
# flavour-asymmetric: there is no synchronous form of it to register, in this
# distribution or any other, so the synchronous registry alone carries it.
# The asynchronous side carries no mark because it is waiting on a class that
# does not exist yet rather than on an import -- ``SemanticSignal`` is
# unwritten, here and everywhere, so asking the asynchronous registry for it
# today reports an unknown kind. That is the gap this mechanism exists to
# close, and it closes when the rung ships: the mark belongs beside the class,
# and inventing one now would name an install that supplies nothing.
# ``authority`` has both flavours and ships in another distribution, so both
# registries carry it and both marks are cleared the moment an application
# imports the module that implements it. A consumer who writes either kind and
# registers their own clears the mark the same way -- which is the extension
# point, not a leak.
signal_backends.declare_unavailable(
    "semantic",
    reason="SemanticSignal has no synchronous form",
    metadata={"flavour": "async", "needs_io": True},
)

# The reason names the distribution **and** the import, because they answer
# two different questions and a reader stuck on this key has both: what do I
# install, and -- having installed it -- why is the kind still unknown. The
# second is the one a package name alone leaves open, since the registration
# happens at a module's import rather than at the distribution's presence.
_SHIPS_IN_XIZATION = (
    "AuthoritySignal ships in dataknobs-xization; import "
    "dataknobs_xization.entity_resolution to register it"
)
_XIZATION_METADATA = {
    "needs_io": False,
    "requires_install": "pip install dataknobs-xization",
}
signal_backends.declare_unavailable(
    "authority",
    reason=_SHIPS_IN_XIZATION,
    metadata=dict(_XIZATION_METADATA, flavour="sync"),
)
async_signal_backends.declare_unavailable(
    "authority",
    reason=_SHIPS_IN_XIZATION,
    metadata=dict(_XIZATION_METADATA, flavour="async"),
)
