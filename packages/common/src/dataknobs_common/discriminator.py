"""Generic discriminator Protocol for value-to-kind classification.

A discriminator classifies an input value into a kind (label, enum
member, or arbitrary tag). Implementations include backend-key
classifiers (e.g. :meth:`KnowledgeResourceBackend.classify_key`),
LLM-backed label classifiers, and composing discriminators that read
multiple fields from a payload.

The Protocol exists so consumer code composing classification logic
across multiple sources can write against one shape instead of N
backend-specific signatures.

Composing reference implementations cover the common option space:

    CallableDiscriminator       — wraps a Callable[[InputT], KindT]
    MappingDiscriminator        — fast lookup against a static Mapping
    MultiFieldDiscriminator     — classifies multiple fields of a payload
    ChainedDiscriminator        — tries each in order; first non-default wins
    AsyncCallableDiscriminator  — async sibling of CallableDiscriminator
    AsyncChainedDiscriminator   — async ChainedDiscriminator; mixes sync + async
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

InputT_contra = TypeVar("InputT_contra", contravariant=True)
KindT_co = TypeVar("KindT_co", covariant=True)
_InputT = TypeVar("_InputT")
_KindT = TypeVar("_KindT")


@runtime_checkable
class Discriminator(Protocol, Generic[InputT_contra, KindT_co]):
    """Synchronous value-to-kind classifier.

    Implementations classify an input value into a kind. The kind type
    is implementation-defined: enum members, strings, dataclasses, etc.

    Implementations MUST be deterministic (same input → same kind on
    repeated calls within a single process) and MUST NOT mutate the
    input value.
    """

    def classify(self, value: InputT_contra) -> KindT_co: ...


@runtime_checkable
class AsyncDiscriminator(Protocol, Generic[InputT_contra, KindT_co]):
    """Asynchronous value-to-kind classifier.

    Used when classification requires I/O (LLM call, database lookup,
    cached resolution). The contract matches :class:`Discriminator`
    otherwise: deterministic, non-mutating.
    """

    async def classify(self, value: InputT_contra) -> KindT_co: ...


# ---------------------------------------------------------------------------
# Composing reference implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallableDiscriminator(Generic[_InputT, _KindT]):
    """Wraps a ``Callable[[InputT], KindT]`` for ad-hoc classifier
    construction.

    Production-ready for cases where the classification logic is a plain
    function (lambdas, module-level functions, partial applications).
    """

    fn: Callable[[_InputT], _KindT]

    def classify(self, value: _InputT) -> _KindT:
        return self.fn(value)


@dataclass(frozen=True, eq=False)
class MappingDiscriminator(Generic[_InputT, _KindT]):
    """Fast lookup against a static mapping.

    Returns ``mapping[value]`` if present, ``default`` otherwise. Useful
    for fixed enumerations where the classifier is a small lookup table.

    The input type must be hashable. ``frozen=True`` prevents accidental
    reassignment of ``mapping`` or ``default`` after construction;
    ``eq=False`` falls back to identity equality / identity hash because
    the typical ``mapping`` value (``dict``) is not hashable — opting
    into dataclass-generated value equality would make
    :class:`MappingDiscriminator` instances raise ``TypeError`` on
    ``hash()``. Identity hashing is the right default since two
    distinct ``MappingDiscriminator`` instances wrapping equal mappings
    are not generally interchangeable from the consumer's perspective.
    """

    mapping: Mapping[_InputT, _KindT]
    default: _KindT

    def classify(self, value: _InputT) -> _KindT:
        return self.mapping.get(value, self.default)


@dataclass(frozen=True)
class MultiFieldDiscriminator:
    """Reads multiple fields from a mapping-shaped payload and classifies
    each via its declared discriminator. Returns a dict of per-field
    kinds.

    Useful for event-routing pipelines that classify multiple aspects of
    a payload (e.g. key kind + intent + tenant) without coupling the
    consumer's dispatch logic to specific backend interfaces.

    Example::

        backend_discriminator = BackendKeyDiscriminator(backend)
        intent_discriminator = CallableDiscriminator(classify_intent)
        multi = MultiFieldDiscriminator({
            "key": backend_discriminator,
            "intent": intent_discriminator,
        })
        result = multi.classify({"key": "content/foo", "intent": "review"})
        # result == {"key": KnowledgeKeyKind.CONTENT, "intent": ...}

    Fields missing from the payload are classified as ``None`` in the
    result dict (not omitted) so consumers can distinguish "missing"
    from "classified as None".

    The result dict iterates in the same order as
    ``field_discriminators``. Python's built-in ``dict`` (and
    :class:`collections.OrderedDict`) iterate in insertion order, so
    the common usage shape (``MultiFieldDiscriminator({...})`` with a
    dict literal) preserves field-declaration order. The annotation is
    intentionally :class:`~collections.abc.Mapping` rather than
    ``dict`` so subclasses, ``MappingProxyType`` wrappers, and other
    mapping types are accepted — but a hand-rolled
    :class:`~collections.abc.Mapping` impl with arbitrary iteration
    order will yield a result dict with unspecified key order. Pass an
    ordered mapping if downstream code depends on it.
    """

    field_discriminators: Mapping[str, Discriminator[Any, Any]]

    def classify(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            field_name: (
                discriminator.classify(value[field_name])
                if field_name in value
                else None
            )
            for field_name, discriminator in self.field_discriminators.items()
        }


@dataclass(frozen=True)
class ChainedDiscriminator(Generic[_InputT, _KindT]):
    """Tries each discriminator in order; first non-default result wins.

    Useful for layered classification: try the cheap rule first, fall
    back to the expensive classifier. ``default`` is the sentinel that
    signals "no match — try the next discriminator."

    All ``inner`` discriminators MUST return the same kind type.

    Example:
        keyword_classifier = MappingDiscriminator({"yes": "accept", "no": "decline"}, default="unknown")
        llm_classifier = CallableDiscriminator(llm_classify)
        chained = ChainedDiscriminator(
            inner=[keyword_classifier, llm_classifier],
            default="unknown",
        )
        chained.classify("ok")  # falls through to llm_classifier
    """

    inner: Sequence[Discriminator[_InputT, _KindT]]
    default: _KindT

    def classify(self, value: _InputT) -> _KindT:
        for discriminator in self.inner:
            result = discriminator.classify(value)
            if result != self.default:
                return result
        return self.default


# ---------------------------------------------------------------------------
# Async composing implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AsyncCallableDiscriminator(Generic[_InputT, _KindT]):
    """Async sibling of :class:`CallableDiscriminator`."""

    fn: Callable[[_InputT], Awaitable[_KindT]]

    async def classify(self, value: _InputT) -> _KindT:
        return await self.fn(value)


@dataclass(frozen=True)
class AsyncChainedDiscriminator(Generic[_InputT, _KindT]):
    """Async sibling of :class:`ChainedDiscriminator`. Accepts mixed
    sync/async inner discriminators; sync ones are called directly, async
    ones awaited.
    """

    inner: Sequence[Any]  # Discriminator OR AsyncDiscriminator
    default: _KindT

    async def classify(self, value: _InputT) -> _KindT:
        for discriminator in self.inner:
            result = discriminator.classify(value)
            if asyncio.iscoroutine(result):
                result = await result
            if result != self.default:
                return result
        return self.default


__all__ = [
    "AsyncCallableDiscriminator",
    "AsyncChainedDiscriminator",
    "AsyncDiscriminator",
    "CallableDiscriminator",
    "ChainedDiscriminator",
    "Discriminator",
    "MappingDiscriminator",
    "MultiFieldDiscriminator",
]
