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

from typing import TYPE_CHECKING

from dataknobs_common.entity_resolution.protocols import AsyncMatchSignal, MatchSignal
from dataknobs_common.entity_resolution.signals import (
    AliasSignal,
    AsyncAliasSignal,
    AsyncExactNormalizedSignal,
    ExactNormalizedSignal,
)
from dataknobs_common.registry import PluginRegistry

if TYPE_CHECKING:
    from typing import Any

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


def _make_async_exact(config: dict[str, Any]) -> AsyncMatchSignal:
    return AsyncExactNormalizedSignal(config["entities"], normalizer=config.get("normalizer"))


def _make_async_alias(config: dict[str, Any]) -> AsyncMatchSignal:
    return AsyncAliasSignal(config["entities"], normalizer=config.get("normalizer"))


#: Every rung declares its flavour and whether it needs I/O, so a door can
#: refuse a composition *before* building anything. A capability read off an
#: instance is read too late to refuse with.
_DECLARED_METADATA = {"flavour": "sync", "needs_io": False}
_ASYNC_DECLARED_METADATA = {"flavour": "async", "needs_io": False}

signal_backends.register("exact", _make_exact, metadata=_DECLARED_METADATA)
signal_backends.register("alias", _make_alias, metadata=_DECLARED_METADATA)
async_signal_backends.register("exact", _make_async_exact, metadata=_ASYNC_DECLARED_METADATA)
async_signal_backends.register("alias", _make_async_alias, metadata=_ASYNC_DECLARED_METADATA)


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
# One member today, and that is a measurement rather than a shape: of the
# rungs the design contemplates, only this one is flavour-asymmetric. A
# consumer who writes a synchronous rung under this kind registers it, and
# registering clears the mark -- which is the extension point, not a leak.
signal_backends.declare_unavailable(
    "semantic",
    reason="SemanticSignal has no synchronous form",
    metadata={"flavour": "async", "needs_io": True},
)
