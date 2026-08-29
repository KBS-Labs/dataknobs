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
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..records import Record

#: What both classes joined on before either could say so.
DEFAULT_FIELD_SEPARATOR = " "

#: The digest of the assembled text.
CONTENT_HASH_KEY = "content_hash"

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


def recompute_content_hash(
    record: Record,
    metadata: dict[str, Any] | None,
    fallback_source_fields: Sequence[str],
    fallback_separator: str = DEFAULT_FIELD_SEPARATOR,
) -> str | None:
    """Reproduce a stored digest from the record's current values.

    Reads the assembly description out of ``metadata`` where it is present,
    and falls back to the caller's own configuration where it is not — which
    is the case for every record written before that description was stored.

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
    metadata = metadata or {}

    source_fields = metadata.get(SOURCE_FIELDS_KEY) or fallback_source_fields
    if not source_fields:
        return None

    # An empty string is a legitimate separator, so absence is the only
    # thing that may fall back.
    separator = metadata.get(FIELD_SEPARATOR_KEY)
    if separator is None:
        separator = fallback_separator

    text = assemble_source_text(record, source_fields, separator)
    if not text:
        return None
    return compute_content_hash(text)
