# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Building a callable a configuration names, by one rule.

A document cannot write a callable, so it writes a name, and a
:class:`~dataknobs_common.registry.PluginRegistry` of factories turns the name
into one. :data:`~dataknobs_common.text.TOKENIZERS` and
:data:`~dataknobs_common.text.VARIANT_RULES` are two such registries, and both
builders follow the same steps: read the name from a string or a mapping, refuse
a name that cannot be built here, build it, and refuse a factory whose result is
not callable. :func:`built_from_config` is those steps, written once, so the
next named seam is a :class:`NamedSeam` rather than a third copy.

**What a failing factory's caller sees is decided here, once.** A factory's
words are repeated only when dataknobs wrote them; a consumer factory's text can
carry the path or URL it was handed, so its error is named by type and kept
under ``__cause__``. Either way the caller catches one type. ``create`` raises a
factory's :class:`~dataknobs_common.exceptions.NotFoundError` as a
``NotFoundError`` rather than an ``OperationError``; the name was found
registered before anything was built, so that too is the factory's failure and
not a miss on the name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dataknobs_common.exceptions import (
    NotFoundError,
    OperationError,
    ValidationError,
    _given,
    _given_value,
    _written_by_dataknobs,
)
from dataknobs_common.registry import PluginRegistry, _shown_failure

__all__ = ["NamedSeam", "built_from_config"]


@dataclass(frozen=True, eq=False)
class NamedSeam:
    """One registry of named factories, and the words its refusals use.

    Compared and hashed by identity: a seam is declared once per registry.

    Attributes:
        registry: The factories, by name.
        key: The mapping key a spec names one under (``name``, ``rule``).
        noun: What one is called in a message (``tokenizer``,
            ``variant rule``).
        plural: What the registered ones are called when they are listed
            (``tokenizers``, ``rules``).
        result: What a factory returns (``tokenizer``, ``rule``).
        shape: What a well-formed spec is, completing "must be ...".
        context_key: The context key a refusal records the name under.
    """

    registry: PluginRegistry[Any]
    key: str
    noun: str
    plural: str
    result: str
    shape: str
    context_key: str


def built_from_config(
    seam: NamedSeam,
    spec: object,
    *,
    subject: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> tuple[str, Any]:
    """Build what ``spec`` names in ``seam``'s registry, or refuse by name.

    Args:
        seam: The registry and its words.
        spec: A name, or a mapping naming one under ``seam.key`` with the
            factory's parameters beside it.
        subject: How a malformed spec is referred to; ``seam.noun`` when not
            given. A list of specs passes ``"variant rule 2"``.
        context: Carried into every refusal's context, such as a position.

    Returns:
        The name and what its factory built.

    Raises:
        ValidationError: For a spec of any other shape, a name the registry
            does not hold or holds as unavailable, a factory that fails, or
            one that returns something not callable. An unavailable name's
            reason is in the message, and in the context as ``unavailable``.
    """
    extra = dict(context or {})
    config = {seam.key: spec} if isinstance(spec, str) else spec
    name = config.get(seam.key) if isinstance(config, Mapping) else None
    if not isinstance(config, Mapping) or not isinstance(name, str) or not name:
        raise _written_by_dataknobs(
            ValidationError(
                f"{subject or seam.noun} must be {seam.shape}; given {_given(spec)}",
                context={**extra, "spec": _given_value(spec)},
            )
        )
    named = {**extra, seam.context_key: name}
    if not seam.registry.is_registered(name):
        unavailable = seam.registry.unavailable_reason(name)
        if unavailable is not None:
            # Spelled right, so "unknown" would send its author after a typo.
            raise _written_by_dataknobs(
                ValidationError(
                    f"{seam.noun} {name!r} is not available here: {unavailable}",
                    context={**named, "unavailable": unavailable},
                )
            )
        raise _written_by_dataknobs(
            ValidationError(
                f"unknown {seam.noun} {name!r}. Registered {seam.plural}: "
                f"{sorted(seam.registry.list_keys())}",
                context=named,
            )
        )
    try:
        built = seam.registry.create(config=dict(config))
    except (NotFoundError, OperationError) as error:
        # A consumer factory's text may carry what it was handed, so unless
        # dataknobs wrote it, it is named by type and kept under ``__cause__``.
        shown = _shown_failure(error)
        if shown is None:
            # Names only the seam's name and the error's type, so it is ours.
            raise _written_by_dataknobs(
                ValidationError(
                    f"{seam.noun} {name!r} could not be built ({type(error.__cause__).__name__})",
                    context=named,
                )
            ) from error
        raise _written_by_dataknobs(
            ValidationError(f"{seam.noun} {name!r} {shown}", context=named)
        ) from error
    if not callable(built):
        raise _written_by_dataknobs(
            ValidationError(
                f"{seam.noun} {name!r}'s factory returned a {type(built).__name__}, "
                f"which is not callable; a factory returns a {seam.result}",
                context=named,
            )
        )
    return name, built
