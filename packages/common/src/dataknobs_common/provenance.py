# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Where a piece of content came from, and what has been done to it since.

Five records describe it: a :class:`SourceReference` for each source it rests
on, a :class:`ToolInvocation` for each tool used, an :class:`LLMInvocation` for
each model call, a :class:`RevisionRecord` for each revision, and the
:class:`ProvenanceRecord` that holds them all. :func:`create_provenance` makes
one, and :func:`now_iso` is the timestamp every record defaults to.

They are plain data and need nothing else from dataknobs, so a consumer that
writes text with a model can record its provenance without running bots.
``dataknobs_bots.artifacts.provenance`` re-exports these same names, and an
artifact's provenance is a :class:`ProvenanceRecord`.

**The records are frozen.** A record handed to a store, a renderer or another
task cannot change under it:

* each field that holds several values keeps a **tuple** of whatever iterable
  it was given, a list included;
* a mapping field (:attr:`ProvenanceRecord.creation_context`,
  :attr:`ToolInvocation.parameters`) keeps **a frozen copy** of the mapping it
  was given, a :class:`~dataknobs_common.copying.FrozenMapping`: the caller's
  dict can go on changing without reaching the record, and the record's own
  mapping cannot be changed by whoever reads it. Nested dicts become
  ``FrozenMapping`` too and nested lists tuples; other values are shared, as
  :func:`~dataknobs_common.copying.freeze_structure` shares them, and a
  mapping that holds itself is refused with ``ValueError``;
* ``to_dict`` returns new, plain containers (dicts and lists), so editing what
  it returns does not reach the record either.

The three records holding no mapping (:class:`SourceReference`,
:class:`LLMInvocation`, :class:`RevisionRecord`) are hashable, so a set of
sources works. The two that hold one (:class:`ToolInvocation`,
:class:`ProvenanceRecord`) are not, as a ``FrozenMapping`` is not:
``hash()`` raises ``TypeError`` and :class:`collections.abc.Hashable`
answers no.

So a revision is a new record, made with :func:`dataclasses.replace`:

```python
import dataclasses

from dataknobs_common.provenance import (
    RevisionRecord,
    SourceReference,
    create_provenance,
)

record = create_provenance(
    "system:generator:summary_v1",
    "generator",
    sources=[SourceReference(source_id="QUOTE-0117", source_type="row")],
)
revised = dataclasses.replace(
    record,
    revision_history=(
        *record.revision_history,
        RevisionRecord(reason="edited", triggered_by="user:buyer"),
    ),
)
assert record.revision_history == ()   # the record handed on is unchanged
```

**Serialised form.** ``to_dict`` writes JSON-shaped data and ``from_dict``
reads it back; an optional field that is unset is left out. Three fields
describe a part of the content rather than the whole, and each is written only
when set, so a record that uses none of them serialises exactly as it did
before they existed:

* :attr:`LLMInvocation.invocation_id`, so a span of text can name the call that
  wrote it;
* :attr:`RevisionRecord.part`, :attr:`~RevisionRecord.after` and
  :attr:`~RevisionRecord.sources`: which part a revision changed, what that
  part became, and which sources it kept.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from dataknobs_common.copying import FrozenMapping, freeze_structure, thaw_structure

__all__ = [
    "LLMInvocation",
    "ProvenanceRecord",
    "RevisionRecord",
    "SourceReference",
    "ToolInvocation",
    "create_provenance",
    "now_iso",
]


def now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def _revision_id() -> str:
    """A new revision's id: ``rev_`` and 12 hex digits."""
    return f"rev_{uuid.uuid4().hex[:12]}"


def _tuple_of(values: Iterable[Any], owner: str, name: str) -> tuple[Any, ...]:
    """A tuple of ``values``, refusing a single string given in their place.

    A string is an iterable of its characters, so ``review_history="eval_1"``
    would otherwise be held as six one-letter ids.
    """
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{owner}: {name} takes an iterable of values, "
            f"not a single {type(values).__name__}: {values!r}"
        )
    return tuple(values)


def _hold_tuples(record: object, *names: str) -> None:
    """Keep each named field of a frozen ``record`` as a tuple of what it was given."""
    for name in names:
        held = _tuple_of(getattr(record, name), type(record).__name__, name)
        object.__setattr__(record, name, held)


def _hold_frozen(record: object, name: str) -> None:
    """Keep the named mapping field of a frozen ``record`` as a frozen copy."""
    given = getattr(record, name)
    if not isinstance(given, Mapping):
        raise TypeError(
            f"{type(record).__name__}.{name} takes a mapping, not {type(given).__name__}: {given!r}"
        )
    object.__setattr__(record, name, freeze_structure(given))


@dataclass(frozen=True)
class SourceReference:
    """A source used in making the content.

    Attributes:
        source_id: ID of the source artifact, document, row or external
            reference.
        source_type: Category of source (``artifact``, ``document``,
            ``vector_result``, ``user_input``, ``external``, ...).
        source_location: URI, file path, corpus id or vector store reference.
        relevance: Why this source was used.
        excerpt: The part of the source used.
        confidence: Relevance confidence score (0.0 to 1.0).
    """

    source_id: str
    source_type: str
    source_location: str | None = None
    relevance: str = ""
    excerpt: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary, leaving out unset optional fields."""
        result: dict[str, Any] = {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "confidence": self.confidence,
        }
        if self.source_location is not None:
            result["source_location"] = self.source_location
        if self.relevance:
            result["relevance"] = self.relevance
        if self.excerpt is not None:
            result["excerpt"] = self.excerpt
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SourceReference:
        """Deserialize from a dictionary written by :meth:`to_dict`."""
        return cls(
            source_id=data["source_id"],
            source_type=data["source_type"],
            source_location=data.get("source_location"),
            relevance=data.get("relevance", ""),
            excerpt=data.get("excerpt"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass(frozen=True)
class ToolInvocation:
    """A tool used in making the content.

    Attributes:
        tool_name: Function or tool identifier.
        tool_version: Version of the tool, if applicable.
        parameters: Input parameters passed to the tool. Any mapping is
            accepted, and the record keeps a frozen copy of it
            (a :class:`~dataknobs_common.copying.FrozenMapping`).
        timestamp: ISO 8601 timestamp of the invocation.
    """

    tool_name: str
    tool_version: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=FrozenMapping)
    timestamp: str = field(default_factory=now_iso)

    def __post_init__(self) -> None:
        _hold_frozen(self, "parameters")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary, leaving out an unset ``tool_version``."""
        result: dict[str, Any] = {
            "tool_name": self.tool_name,
            "parameters": thaw_structure(self.parameters),
            "timestamp": self.timestamp,
        }
        if self.tool_version is not None:
            result["tool_version"] = self.tool_version
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolInvocation:
        """Deserialize from a dictionary written by :meth:`to_dict`."""
        return cls(
            tool_name=data["tool_name"],
            tool_version=data.get("tool_version"),
            parameters=data.get("parameters", {}),
            timestamp=data.get("timestamp", ""),
        )


# Declared after the class body, as ``FrozenMapping.__hash__`` is: the record
# holds a mapping, which cannot be hashed, so it says so rather than offering
# a hash that raises.
ToolInvocation.__hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LLMInvocation:
    """A model call made in making the content.

    Attributes:
        purpose: Why the model was called (``decode_intent``, ``write``,
            ``extract``, ...).
        model: Model identifier used.
        prompt_hash: Hash of the prompt for reproducibility (not the full
            prompt, for privacy).
        timestamp: ISO 8601 timestamp of the invocation.
        invocation_id: An id a span of the content can name to say this call
            wrote it. Optional; written by :meth:`to_dict` only when set.
    """

    purpose: str
    model: str = ""
    prompt_hash: str = ""
    timestamp: str = field(default_factory=now_iso)
    invocation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary; ``invocation_id`` only when set."""
        result: dict[str, Any] = {
            "purpose": self.purpose,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "timestamp": self.timestamp,
        }
        if self.invocation_id:
            result["invocation_id"] = self.invocation_id
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LLMInvocation:
        """Deserialize from a dictionary written by :meth:`to_dict`."""
        return cls(
            purpose=data["purpose"],
            model=data.get("model", ""),
            prompt_hash=data.get("prompt_hash", ""),
            timestamp=data.get("timestamp", ""),
            invocation_id=data.get("invocation_id", ""),
        )


@dataclass(frozen=True)
class RevisionRecord:
    """One revision: of the whole content, or of one part of it.

    Attributes:
        revision_id: Unique identifier for this revision. Generated
            (``rev_`` and 12 hex digits) when not given, both here and in
            :meth:`from_dict`; a caller that wants the same decisions to give
            the same record passes its own.
        previous_version: The version, or the text, being revised.
        reason: Why the revision was made.
        changes_summary: What changed in this revision.
        triggered_by: Who or what triggered the revision
            (``rubric_evaluation:eval_123``, ``user:jane``, ``system``).
        timestamp: ISO 8601 timestamp.
        part: Which part was revised (``line-2``); empty for the whole.
            Written by :meth:`to_dict` only when set.
        after: The part's text after the revision: ``""`` where the part was
            removed, ``None`` where it was not recorded. Written by
            :meth:`to_dict` only when not ``None``.
        sources: IDs of the sources the revised part kept. Any iterable is
            accepted and held as a tuple. Written by :meth:`to_dict` only
            when non-empty.
    """

    revision_id: str = field(default_factory=_revision_id)
    previous_version: str = ""
    reason: str = ""
    changes_summary: str = ""
    triggered_by: str = ""
    timestamp: str = field(default_factory=now_iso)
    part: str = ""
    after: str | None = None
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _hold_tuples(self, "sources")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary; each part-level field only when set."""
        result: dict[str, Any] = {
            "revision_id": self.revision_id,
            "previous_version": self.previous_version,
            "reason": self.reason,
            "changes_summary": self.changes_summary,
            "triggered_by": self.triggered_by,
            "timestamp": self.timestamp,
        }
        if self.part:
            result["part"] = self.part
        if self.after is not None:
            result["after"] = self.after
        if self.sources:
            result["sources"] = list(self.sources)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RevisionRecord:
        """Deserialize from a dictionary written by :meth:`to_dict`."""
        return cls(
            revision_id=data["revision_id"] if "revision_id" in data else _revision_id(),
            previous_version=data.get("previous_version", ""),
            reason=data.get("reason", ""),
            changes_summary=data.get("changes_summary", ""),
            triggered_by=data.get("triggered_by", ""),
            timestamp=data.get("timestamp", ""),
            part=data.get("part", ""),
            after=data.get("after"),
            sources=tuple(data.get("sources", ())),
        )


@dataclass(frozen=True)
class ProvenanceRecord:
    """Complete provenance for a piece of content: how it was made, and since.

    Attributes:
        created_by: Who or what created it
            (``system:generator:quiz_gen_v1``, ``user:jane``, ``llm:<model>``).
        created_at: ISO 8601 timestamp of creation.
        creation_method: How it was created (``generator``, ``wizard``,
            ``manual``, ``derived``, ``llm_assisted``, ...).
        creation_context: Parameters, configuration and other context. Any
            mapping is accepted, and the record keeps a frozen copy of it
            (a :class:`~dataknobs_common.copying.FrozenMapping`).
        sources: References to the sources used.
        tool_chain: Tools used during creation.
        llm_invocations: Model calls made during creation.
        review_history: IDs of evaluations of the content.
        revision_history: Revisions made since.

    The last five take any iterable and hold a tuple of it. To add to one,
    make a new record with :func:`dataclasses.replace`.
    """

    created_by: str = ""
    created_at: str = field(default_factory=now_iso)
    creation_method: str = ""
    creation_context: Mapping[str, Any] = field(default_factory=FrozenMapping)
    sources: tuple[SourceReference, ...] = ()
    tool_chain: tuple[ToolInvocation, ...] = ()
    llm_invocations: tuple[LLMInvocation, ...] = ()
    review_history: tuple[str, ...] = ()
    revision_history: tuple[RevisionRecord, ...] = ()

    def __post_init__(self) -> None:
        _hold_frozen(self, "creation_context")
        _hold_tuples(
            self,
            "sources",
            "tool_chain",
            "llm_invocations",
            "review_history",
            "revision_history",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary, each nested record by its own ``to_dict``."""
        return {
            "created_by": self.created_by,
            "created_at": self.created_at,
            "creation_method": self.creation_method,
            "creation_context": thaw_structure(self.creation_context),
            "sources": [source.to_dict() for source in self.sources],
            "tool_chain": [tool.to_dict() for tool in self.tool_chain],
            "llm_invocations": [call.to_dict() for call in self.llm_invocations],
            "review_history": list(self.review_history),
            "revision_history": [revision.to_dict() for revision in self.revision_history],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProvenanceRecord:
        """Deserialize from a dictionary written by :meth:`to_dict`."""
        return cls(
            created_by=data.get("created_by", ""),
            created_at=data.get("created_at", ""),
            creation_method=data.get("creation_method", ""),
            creation_context=data.get("creation_context", {}),
            sources=tuple(SourceReference.from_dict(s) for s in data.get("sources", ())),
            tool_chain=tuple(ToolInvocation.from_dict(t) for t in data.get("tool_chain", ())),
            llm_invocations=tuple(
                LLMInvocation.from_dict(c) for c in data.get("llm_invocations", ())
            ),
            review_history=tuple(data.get("review_history", ())),
            revision_history=tuple(
                RevisionRecord.from_dict(r) for r in data.get("revision_history", ())
            ),
        )


# Declared after the class body, as ``FrozenMapping.__hash__`` is: the record
# holds a mapping, which cannot be hashed, so it says so rather than offering
# a hash that raises.
ProvenanceRecord.__hash__ = None  # type: ignore[assignment]


def create_provenance(
    created_by: str,
    creation_method: str,
    *,
    created_at: str | None = None,
    creation_context: Mapping[str, Any] | None = None,
    sources: Iterable[SourceReference] = (),
    tool_chain: Iterable[ToolInvocation] = (),
    llm_invocations: Iterable[LLMInvocation] = (),
    review_history: Iterable[str] = (),
    revision_history: Iterable[RevisionRecord] = (),
) -> ProvenanceRecord:
    """Make a :class:`ProvenanceRecord` naming who or what made the content, and how.

    Every other field is optional. Those holding several values take any
    iterable (a list built up beforehand, a generator) and the record keeps a
    tuple of it; ``creation_context`` takes any mapping and the record keeps a
    frozen copy.

    Args:
        created_by: Who or what created the content.
        creation_method: How it was created (``generator``, ``wizard``,
            ``manual``, ...).
        created_at: ISO 8601 timestamp; now, in UTC, when not given.
        creation_context: Parameters, configuration and other context.
        sources: References to the sources used.
        tool_chain: Tools used during creation.
        llm_invocations: Model calls made during creation.
        review_history: IDs of evaluations of the content.
        revision_history: Revisions made since.

    Returns:
        A new, frozen ProvenanceRecord.
    """
    return ProvenanceRecord(
        created_by=created_by,
        created_at=now_iso() if created_at is None else created_at,
        creation_method=creation_method,
        creation_context=FrozenMapping() if creation_context is None else creation_context,
        sources=tuple(sources),
        tool_chain=tuple(tool_chain),
        llm_invocations=tuple(llm_invocations),
        review_history=_tuple_of(review_history, "create_provenance", "review_history"),
        revision_history=tuple(revision_history),
    )
