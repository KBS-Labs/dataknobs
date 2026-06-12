"""Generic discriminator Protocol for value-to-kind classification.

A discriminator classifies an input value into a kind (label, enum
member, or arbitrary tag). Implementations include backend-key
classifiers (e.g. :meth:`KnowledgeResourceBackend.classify_key`),
LLM-backed label classifiers, and composing discriminators that read
multiple fields from a payload.

The Protocol exists so consumer code composing classification logic
across multiple sources can write against one shape instead of N
backend-specific signatures.
"""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar, runtime_checkable

InputT_contra = TypeVar("InputT_contra", contravariant=True)
KindT_co = TypeVar("KindT_co", covariant=True)


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


__all__ = [
    "AsyncDiscriminator",
    "Discriminator",
]
