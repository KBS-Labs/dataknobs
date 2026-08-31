"""How source fields become the text a vector was built from.

A vector is stale when the text that produced it is no longer the text the
record would produce now. Answering that requires assembling the embedder's
input, and two classes need to assemble it: :class:`VectorTextSynchronizer`,
which embeds, and :class:`ChangeTracker`, which decides what to queue. They
built it separately and drifted — identical loops joined on different
separators — so a corpus synced with any non-default separator was reported
permanently outdated.

The assembly lives here, once, and the digest is taken over its output. Two
consequences are worth stating because they are the reason this module exists
rather than a shared ``_hash`` helper:

* **The digest covers exactly the string that was embedded.** Not the field
  values, not a canonical join of them — the bytes the embedder saw. That is
  what makes "re-embed if and only if the input changed" true with no false
  positives and no false negatives.
* **The assembly is described in the stored metadata**, so a reader
  reproduces it from the record rather than from its own configuration. A
  tracker cannot disagree with a synchronizer about a separator it was never
  told, because it does not use the one it was given.

Records written before that description existed carry no such keys. Callers
pass their own configuration as the fallback, which is what those records were
digested under, so no stored hash is invalidated and nothing re-embeds on
upgrade.

Two questions, two functions
----------------------------

The description above is right for a *reader* and wrong for a *writer*, and
collapsing them into one function is a defect rather than a simplification:

============================  ==================================  ==================
Class                         Question                            Authority
============================  ==================================  ==================
:class:`ChangeTracker`        "has the source changed since this   the **record**
                              was embedded?"
:class:`VectorTextSynchronizer` "would I produce a different       **its own**
                              string now?"                         configuration
============================  ==================================  ==================

A synchronizer that deferred to the record could never notice its own
configuration changing: re-point ``text_fields`` or change ``field_separator``
and every record would keep matching the assembly it was written under, so
``sync_all()`` would report nothing to do and the new configuration would never
take effect. :func:`current_content_hash` is the writer's question and consults
no metadata; :func:`recompute_content_hash` is the reader's and prefers what
the record carries.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..records import Record

logger = logging.getLogger(__name__)

#: What both classes joined on before either could say so.
DEFAULT_FIELD_SEPARATOR = " "

#: The digest of the assembled text.
CONTENT_HASH_KEY = "content_hash"

#: The model that produced the vector. The second key of the staleness
#: contract: the digest answers whether the TEXT changed, this answers
#: whether the MODEL did, and a vector is current only if both agree.
#: Published here rather than spelled at each site because it was spelled
#: at each site, and a reader reaching for the wrong one gets silence --- a
#: key nothing wrote reads as absent, which every reader treats as
#: "unknown, assume current".
MODEL_NAME_KEY = "model_name"

#: The field names that were assembled, in order.
SOURCE_FIELDS_KEY = "content_source_fields"

#: The separator they were joined on.
FIELD_SEPARATOR_KEY = "content_field_separator"


def assemble_source_text(
    record: Record,
    source_fields: Sequence[str],
    separator: str = DEFAULT_FIELD_SEPARATOR,
) -> str:
    """Build the text a vector over ``source_fields`` is derived from.

    Falsy values are dropped rather than joined as empty strings, which is
    what both callers did independently and what the stored digests were
    computed under.

    Args:
        record: The record to read source values from.
        source_fields: Field names to assemble, in order.
        separator: What to join them on.

    Returns:
        The assembled text, empty if no source field held a value.
    """
    parts = []
    for field_name in source_fields:
        value = record.get_value(field_name)
        if value:
            parts.append(str(value))
    return separator.join(parts)


def compute_content_hash(content: str) -> str:
    """Digest assembled text for change detection.

    Not a security primitive — this answers "is this the same string as last
    time", and md5 is what the stored digests were computed with.
    """
    return hashlib.md5(content.encode()).hexdigest()


def content_hash_metadata(
    source_fields: Sequence[str],
    separator: str,
    content_hash: str,
) -> dict[str, Any]:
    """Describe an assembly completely enough for another class to repeat it.

    Args:
        source_fields: The field names that were assembled, in order.
        separator: What they were joined on.
        content_hash: The digest of the result.

    Returns:
        Metadata to store on the vector field beside its value.
    """
    return {
        CONTENT_HASH_KEY: content_hash,
        SOURCE_FIELDS_KEY: list(source_fields),
        FIELD_SEPARATOR_KEY: separator,
    }


def stored_assembly(
    metadata: dict[str, Any] | None,
) -> tuple[list[str] | None, str | None]:
    """Read whatever assembly description a record carries.

    The two halves are independent: a record may name its fields without
    naming its separator, and the caller falls back per key rather than
    discarding a description because half of it is missing.

    This crosses a persistence trust boundary — the values come back from
    whatever store wrote them and are not guaranteed to be the shapes that
    were written. A half that is not usable is reported as absent, which puts
    the caller in the same position as for a record written before
    descriptions existed: it falls back to its own configuration.

    Args:
        metadata: The vector field's metadata, if any.

    Returns:
        ``(source_fields, separator)``, either of which is ``None`` when the
        record does not usably say.
    """
    if not metadata:
        return None, None

    source_fields: list[str] | None = None
    stored_fields = metadata.get(SOURCE_FIELDS_KEY)
    if stored_fields:
        if isinstance(stored_fields, (list, tuple)) and all(
            isinstance(name, str) for name in stored_fields
        ):
            source_fields = list(stored_fields)
        else:
            logger.warning(
                "Ignoring stored %s: expected a list of field names, got %r",
                SOURCE_FIELDS_KEY,
                stored_fields,
            )

    # An empty string is a legitimate separator, so absence is the only thing
    # that may fall back -- absence being the key missing, not a falsy value.
    separator: str | None = None
    stored_separator = metadata.get(FIELD_SEPARATOR_KEY)
    if stored_separator is not None:
        if isinstance(stored_separator, str):
            separator = stored_separator
        else:
            logger.warning(
                "Ignoring stored %s: expected a string, got %r",
                FIELD_SEPARATOR_KEY,
                stored_separator,
            )

    return source_fields, separator


def describes_its_assembly(metadata: dict[str, Any] | None) -> bool:
    """Whether a reader can reproduce this vector's text without being told.

    Both halves have to be present: field names alone leave a reader guessing
    the separator, which is the disagreement this description exists to end.
    """
    source_fields, separator = stored_assembly(metadata)
    return source_fields is not None and separator is not None


def current_content_hash(
    record: Record,
    source_fields: Sequence[str],
    separator: str = DEFAULT_FIELD_SEPARATOR,
) -> str | None:
    """Digest the text *this* configuration would feed the embedder now.

    The writer's question. It consults no stored metadata, because a class
    that maintains a vector field is the authority on how that field is
    assembled — deferring to the record would make the class's own
    configuration unchangeable, its every edit invisible to the sweep that
    is supposed to apply it.

    Args:
        record: The record to read current source values from.
        source_fields: The fields this caller assembles, in order.
        separator: What this caller joins them on.

    Returns:
        The digest of the current text, or ``None`` when there is no text to
        digest — no source fields to read, or none of them holding a value.
    """
    if not source_fields:
        return None
    text = assemble_source_text(record, source_fields, separator)
    if not text:
        return None
    return compute_content_hash(text)


def recompute_content_hash(
    record: Record,
    metadata: dict[str, Any] | None,
    fallback_source_fields: Sequence[str],
    fallback_separator: str = DEFAULT_FIELD_SEPARATOR,
) -> str | None:
    """Reproduce a stored digest from the record's current values.

    The reader's question, for a class that did not write the vector and so
    has no standing to impose its own assembly on it. Reads the description
    out of ``metadata`` where it is present, and falls back to the caller's
    own configuration where it is not — which is the case for every record
    written before that description was stored.

    A writer deciding whether to re-embed wants :func:`current_content_hash`
    instead; see the module docstring for why the two cannot share an answer.

    Args:
        record: The record to read current source values from.
        metadata: The vector field's metadata, if any.
        fallback_source_fields: Fields to assemble when the metadata does not
            say, i.e. the reading class's own configuration.
        fallback_separator: Separator to use when the metadata does not say.

    Returns:
        The digest of the current text, or ``None`` when there is no text to
        digest — no source fields to read, or none of them holding a value.
    """
    stored_fields, stored_separator = stored_assembly(metadata)

    source_fields = stored_fields if stored_fields is not None else list(fallback_source_fields)
    separator = stored_separator if stored_separator is not None else fallback_separator

    return current_content_hash(record, source_fields, separator)
