"""Bot-flavored vocabulary for :mod:`dataknobs_common.packs`.

A *behavior pack* is a named, frozen bundle of bot-shaping declarations —
middleware to install, a reasoning strategy to require, stage primitives to
expect. A deployment selects packs and tunes them in one binding block, and
resolution folds the selection into a single composed declaration under the
per-field rules :class:`BehaviorPackSpec` declares.

**DataKnobs ships zero packs.** This module supplies the vocabulary and the
composition rules; the pack *content* is a deployment's own policy. There is
no module-level registry either — a pack binding is a per-deployment
decision, so a process-global registry would be a multi-tenant hazard.
Construct one and own it::

    from dataknobs_bots import BehaviorPackSpec, BehaviorPackRegistry
    from dataknobs_common.packs import PackRegistry

    registry: BehaviorPackRegistry = PackRegistry(
        "behavior_packs", BehaviorPackSpec
    )
    registry.register_pack(
        BehaviorPackSpec(
            name="audit",
            priority=10,
            middleware=({"class": "acme.mw.AuditMiddleware"},),
        )
    )

The whole install rail, end to end::

    resolution = registry.resolve(config.get("behavior_packs", {}))
    verify_stage_synthesizers(resolution.spec.stage_synthesizers)

    bot = await DynaBot.from_config(
        bot_config,
        platform_middleware=build_middleware(resolution.spec.middleware),
        platform_conversation_middleware=build_conversation_middleware(
            resolution.spec.conversation_middleware
        ),
    )

``required_strategy`` and ``strategy_overrides`` are deliberately left for
the caller to apply to its own reasoning block. No DK-owned build path is
assumed: a deployment that assembles bot config itself keeps doing so, and
reads the composed values as data.

Why each field composes the way it does:

===========================  =================  ==========================
Field                        Rule               Rationale
===========================  =================  ==========================
``required_strategy``        ``UNANIMOUS``      Two packs demanding
                                                different strategies is
                                                unsatisfiable, not
                                                resolvable — silently
                                                keeping one would ship a
                                                bot that violates a pack's
                                                stated requirement.
``strategy_overrides``       ``MERGE``          Independent knobs; the
                                                higher-priority pack wins
                                                a contested key and the
                                                collision is reported.
``middleware``               ``CONCAT``         Order is behavior, and a
                                                repeated spec is a
                                                deliberate second
                                                installation.
``conversation_middleware``  ``CONCAT``         Same, one layer down.
``stage_synthesizers``       ``CONCAT_UNIQUE``  Names, not instances:
                                                registration is
                                                idempotent, so a duplicate
                                                is noise rather than
                                                intent.
===========================  =================  ==========================

Ordering is ascending priority (lower value folds first), matching
:class:`~dataknobs_common.callbacks.PriorityOrdering`. For ``CONCAT``
fields that means the lowest-priority pack's middleware runs first.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.packs import MergeKind, PackRegistry, PackSpec

from .reasoning.stage_synthesizers import stage_synthesizer_backends

__all__ = [
    "BehaviorPackRegistry",
    "BehaviorPackSpec",
    "verify_stage_synthesizers",
]


@dataclass(frozen=True)
class BehaviorPackSpec(PackSpec):
    """A named bundle of bot-shaping declarations.

    Every field is optional — a pack is a *partial* contribution, and an
    unset field is one the pack does not speak to. Fields left at their
    default do not participate in the fold, so a pack that declares only
    ``middleware`` cannot clobber another pack's ``required_strategy``.

    Attributes:
        required_strategy: Reasoning-strategy name this pack requires
            (e.g. ``"wizard"``). An opaque string — this module does not
            resolve it, and the caller decides how to apply it.
        strategy_overrides: Reasoning-block settings the pack contributes.
            Shallow-merged: the higher-priority pack wins a contested key
            and the collision is reported as a ``key_override`` warning.
            A subclass wanting recursive merge declares
            ``dataknobs_config.inheritance.deep_merge`` as this field's
            rule — it matches the :data:`~dataknobs_common.packs.Reducer`
            signature exactly.
        middleware: Raw bot-turn middleware specs, each the
            ``{"class": ..., "params": ..., "optional": ...}`` mapping
            ``DynaBotConfig.middleware`` accepts. Kept opaque so this spec
            and the bot config cannot drift; turn them into instances with
            :func:`~dataknobs_bots.middleware.build_middleware`.
        conversation_middleware: The LLM-call-wrap analogue, for
            :func:`~dataknobs_bots.middleware.build_conversation_middleware`.
            Carried alongside ``middleware`` because a pack that can only
            populate half the install rail is half a pack.
        stage_synthesizers: Names of wizard stage primitives the pack
            expects to be registered. Declaration only — registration stays
            import-time and process-global. Check them with
            :func:`verify_stage_synthesizers`.
    """

    required_strategy: str | None = None
    strategy_overrides: Mapping[str, Any] = field(default_factory=dict)
    middleware: tuple[Mapping[str, Any], ...] = ()
    conversation_middleware: tuple[Mapping[str, Any], ...] = ()
    stage_synthesizers: tuple[str, ...] = ()

    _COMPOSITION = MappingProxyType(
        {
            "required_strategy": MergeKind.UNANIMOUS,
            "strategy_overrides": MergeKind.MERGE,
            "middleware": MergeKind.CONCAT,
            "conversation_middleware": MergeKind.CONCAT,
            "stage_synthesizers": MergeKind.CONCAT_UNIQUE,
        }
    )


#: A :class:`~dataknobs_common.packs.PackRegistry` holding behavior packs.
#: Construct as ``PackRegistry("behavior_packs", BehaviorPackSpec)``; the
#: alias exists so consumer signatures can name the concrete type.
BehaviorPackRegistry: TypeAlias = PackRegistry[BehaviorPackSpec]


def verify_stage_synthesizers(names: Iterable[str]) -> None:
    """Assert that every named stage synthesizer is registered.

    A pack declares synthesizer *names*; the synthesizers themselves are
    registered at import time by whichever module defines them. Nothing
    connects the two, so a typo'd or forgotten name would otherwise surface
    as a wizard stage whose primitive silently never expands.

    This closes that hole without changing the registration model: it
    registers nothing and imports nothing on the caller's behalf. Call it
    after resolution and after importing the modules that register the
    synthesizers you expect.

    Args:
        names: Synthesizer field names to check — typically
            ``resolution.spec.stage_synthesizers``.

    Raises:
        ConfigurationError: If any name is absent from
            :data:`~dataknobs_bots.reasoning.stage_synthesizer_backends`.
            The message lists every missing name and what is registered, so
            one call reports the whole gap rather than the first of it.
    """
    missing = sorted({name for name in names if not stage_synthesizer_backends.has(name)})
    if not missing:
        return

    available = sorted(stage_synthesizer_backends.list_keys())
    raise ConfigurationError(
        f"Stage synthesizers not registered: {missing}. "
        f"Registered: {available or '(none)'}. "
        "A synthesizer is registered by importing the module that calls "
        "register_stage_synthesizer() — check the name and that the module "
        "is imported before this call.",
        context={"missing": missing, "available": available},
    )
