# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Building a callable a configuration names, by one rule.

A document cannot write a callable, so it writes a name, and a
:class:`~dataknobs_common.registry.PluginRegistry` of factories turns the name
into one. :data:`~dataknobs_common.text.TOKENIZERS` and
:data:`~dataknobs_common.text.VARIANT_RULES` are two such registries, and both
builders follow the same steps: read the name from a string or a mapping, refuse
a name nobody registered while listing the names that are, build it, and refuse
a factory whose result is not callable. :func:`built_from_config` is those
steps, written once, so the next named seam is a :class:`NamedSeam` rather than
a third copy.

**What a failing factory's caller sees is decided here, once.** A factory
refusing its parameters raises ``ValueError`` or
:class:`~dataknobs_common.exceptions.ValidationError`, and that text is meant
for the document's author, so it is quoted. Anything else a factory raises is
named by its type alone and chained with ``from``. A factory may load a
dictionary from a path or reach a segmenter over a URL, and its own failure text
can carry either -- the reason ``PluginRegistry`` keeps that text out of its own
error. A :class:`~dataknobs_common.exceptions.NotFoundError` or
:class:`~dataknobs_common.exceptions.OperationError` raised *inside* a factory
reaches ``create`` unwrapped, because the registry reads both as its own. Here
the name has already been found registered, so either one is the factory's
failure and is named by type like any other.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from dataknobs_common.exceptions import NotFoundError, OperationError, ValidationError
from dataknobs_common.registry import PluginRegistry

__all__ = ["NamedSeam", "built_from_config"]

#: What a factory may raise to refuse its parameters; its text is quoted.
_QUOTED: tuple[type[Exception], ...] = (ValueError, ValidationError)


@dataclass(frozen=True)
class NamedSeam:
    """One registry of named factories, and the words its refusals use.

    Attributes:
        registry: The factories, by name.
        key: The mapping key a spec names one under (``name``, ``rule``).
        noun: What one is called in a message (``tokenizer``,
            ``variant rule``).
        plural: What the registered ones are called when they are listed
            (``tokenizers``, ``rules``).
        shape: What a well-formed spec is, completing "must be ...".
        context_key: The context key a refusal records the name under.
    """

    registry: PluginRegistry[Any]
    key: str
    noun: str
    plural: str
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
            does not hold, a factory that fails, or one that returns something
            not callable. A factory refusing its parameters is quoted; any
            other failure is named by its type and chained as the cause.
    """
    extra = dict(context or {})
    config = {seam.key: spec} if isinstance(spec, str) else spec
    name = config.get(seam.key) if isinstance(config, Mapping) else None
    if not isinstance(config, Mapping) or not isinstance(name, str) or not name:
        raise ValidationError(
            f"{subject or seam.noun} must be {seam.shape}; given {spec!r}",
            context={**extra, "spec": spec},
        )
    named = {**extra, seam.context_key: name}
    if not seam.registry.is_registered(name):
        raise ValidationError(
            f"unknown {seam.noun} {name!r}. Registered {seam.plural}: "
            f"{sorted(seam.registry.list_keys())}",
            context=named,
        )
    try:
        built = seam.registry.create(config=dict(config))
    except (NotFoundError, OperationError) as error:
        _refuse_a_failed_factory(seam, name, _factory_failure(error), named)
    if not callable(built):
        raise ValidationError(
            f"{seam.noun} {name!r}'s factory returned {built!r}, which is not "
            f"callable; a factory returns a {seam.noun.split()[-1]}",
            context=named,
        )
    return name, built


def _factory_failure(error: NotFoundError | OperationError) -> BaseException:
    """What the factory itself raised, under ``create``'s wrapping if any.

    ``create`` wraps a factory's exception in an ``OperationError`` carrying it
    on ``__cause__``. An ``OperationError`` or ``NotFoundError`` the factory
    raised itself arrives unwrapped. Wrapped or not, the result is what the
    factory raised.
    """
    if isinstance(error, OperationError) and error.__cause__ is not None:
        return error.__cause__
    return error


def _refuse_a_failed_factory(
    seam: NamedSeam, name: str, failure: BaseException, context: Mapping[str, Any]
) -> NoReturn:
    if isinstance(failure, _QUOTED):
        raise ValidationError(f"{seam.noun} {name!r} {failure}", context=dict(context)) from failure
    raise ValidationError(
        f"{seam.noun} {name!r} could not be built: its factory raised {type(failure).__name__}",
        context={**context, "error_type": type(failure).__name__},
    ) from failure
