# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

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

The other half of the keys on an indexed row
--------------------------------------------

``MODEL_NAME_KEY`` below is one of five keys written onto a single stored row,
and the other four --- which vocabulary, which axis, which node, which surface
forms --- live in ``dataknobs_common.ontology.tags``. The split is by subject
rather than by accident: this one is about the **embedder** that produced the
vector, and is half of the staleness contract the digest above is the other
half of; those four are about **identity**, and belong beside the vocabulary
that mints the ids.

So the two modules name each other, in both directions, because four keys
declared in two modules is otherwise a reader reaching for one module and
finding one of four --- which is this module's own stated defect, one level
up. See ``dataknobs_common/ontology/tags.py``.

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

One rule for the model, and a reader per container
--------------------------------------------------

Publishing ``MODEL_NAME_KEY`` fixed the **key** being spelled at each of its
sites. The rule for comparing what it holds was left at each site, and the
same thing happened again: five readers, and two copies of *absent is unknown,
otherwise exact equality* that had already come to disagree in two cases ---
both times into a false alarm, which is the direction that costs a caller a
corpus re-embed. :func:`is_foreign_model` is that rule now, and every site
calls it.

The shapes it is read out of are genuinely three, and those do not collapse:

=========================  =============================  ======================
Container                  Written by                     Read by
=========================  =============================  ======================
a stored row's metadata    ``add_records``,               :func:`row_model_name`
                           ``bulk_embed_and_store``,
                           ``DedupChecker.register``
a ``{field}_metadata``     ``VectorMetadata.to_dict``     :func:`sidecar_model_name`
sidecar                    (nested), a hand-built dict
                           (flat)
a ``VectorField``          ``VectorField.__init__``       its ``model_name``
                                                          attribute
=========================  =============================  ======================

**A reader accepting every shape everywhere would be a defect rather than a
tolerance.** ``add_records`` writes the flat key onto the row while the same
field's own metadata carries the nested one, in a single call --- so the two
spellings are not alternatives, they are two containers side by side. And a
stored row's metadata is the *caller's* namespace besides: the store documents
five keys of its own and passes everything else through untouched,
``search_similar_records`` promoting each to a field of the record it
synthesises. A corpus of vehicles carries ``model``, and reading that as the
embedding model would warn a correctly built index that its own rankings are
meaningless.

**The comparison in ``dataknobs_data.ontology.registry`` is a fourth site and
deliberately not one of these.** There a configuration *document* is on one
side, so a provider prefix and a version tag are things it may omit and
requiring either would refuse a correctly configured deployment. Here both
sides are ``model_id`` values produced by the same mechanism, where an
omission is two embedders disagreeing. See ``registry._same_model``, which
names this module back.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

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

#: The version of that model, where a writer records one.
#:
#: Published for the reason its sibling above was, and late for the same
#: reason it was: the sibling's publication was prompted by a reader that
#: *"came to read a key nothing wrote"*, and the key that reader was
#: spelling was this one. It stayed a literal at every site afterwards.
#:
#: Written and read only where a version exists to record. A
#: :class:`~dataknobs_data.vector.embedding.TextEmbedder` carries an
#: identity and no version, so the whole embedder seam leaves this absent
#: --- which is why the two keys cannot share a rule: see
#: :func:`sidecar_model_version`.
MODEL_VERSION_KEY = "model_version"

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


def derive_source_text(record: Record, vector_field: str) -> str | None:
    """The text a record's vector was made from, read off the record itself.

    What ``include_source`` always meant. The original design spells it
    *"automatic source retrieval"* --- not records-versus-ids, since the
    record is returned either way, but whether the search result carries the
    text beside the score. No query and no id round-trip is needed: the
    vector field already describes its own assembly, which is the purpose
    :func:`content_hash_metadata` was written for.

    Three answers, in order of how much the record says about itself:

    - it names its fields and separator, so the text is reproduced exactly;
    - it names only the legacy scalar ``source_field``, which is a single
      field name when one was embedded and a comma-joined list when several
      were --- so the lookup succeeds for the first and correctly misses for
      the second, rather than reading a field called ``"title,body"``;
    - it says nothing, which is every vector written before descriptions
      existed. ``None``, gracefully.

    **Past tense, and it is checked.** Reproducing the assembly from the
    record's *current* values gives the text the vector was made from only
    while nothing has edited the record since --- and the vector field already
    carries the answer to that, because :func:`content_hash_metadata` writes
    the digest of the embedded text onto the same dict, in the same call, as
    the field list and the separator. Without the check, updating a title and
    not re-embedding returned a ``source_text`` that provably was not
    embedded, with no signal: a consumer citing it, reranking on it or showing
    it as a snippet is handed text that does not correspond to the vector that
    retrieved it, which is the one failure the parameter exists to prevent.
    The staleness answer is ``None``, which is what this function already
    returns for "the record does not say" --- and a record that has moved on
    since it was embedded does not say.

    A vector carrying no digest is still assembled. Absence of a digest is
    not evidence of staleness, and refusing there would withdraw the
    parameter from every corpus written before the digest existed.

    Args:
        record: The search hit's record.
        vector_field: The field the vector lives on.

    Returns:
        The text the vector was made from, or ``None`` where the record does
        not say --- including where it says the text has changed since.
    """
    vector = record.fields.get(vector_field)
    if vector is None:
        return None

    metadata = getattr(vector, "metadata", None)
    source_fields, separator = stored_assembly(metadata)
    if source_fields:
        # `separator or DEFAULT` would be wrong here: an empty string is a
        # legitimate separator and only absence may fall back, which is the
        # distinction `stored_assembly` reports by returning `None`.
        text = assemble_source_text(
            record,
            source_fields,
            DEFAULT_FIELD_SEPARATOR if separator is None else separator,
        )
        stored_hash = (metadata or {}).get(CONTENT_HASH_KEY)
        if isinstance(stored_hash, str) and compute_content_hash(text) != stored_hash:
            logger.debug(
                "Not deriving source text for %r on record %s: the assembled text "
                "no longer digests to the stored hash, so the vector was made "
                "from something else",
                vector_field,
                record.id,
            )
            return None
        return text

    source_field = getattr(vector, "source_field", None)
    if source_field and source_field in record.fields:
        value = record.get_value(source_field)
        return None if value is None else str(value)

    return None


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


# --------------------------------------------------------------------------
# The model half: one rule, and a reader per container
# --------------------------------------------------------------------------
#
# The digest above answers whether the TEXT changed. What follows answers
# whether the MODEL did, and a stored vector is current only if both agree.
# Which container each reader serves, and why they are not interchangeable,
# is in this module's docstring.
#
# **The nested key itself stays a literal**, in one function below and in
# ``VectorMetadata``/``VectorField``, because it is those dataclasses' own
# serialisation shape rather than a staleness key --- what they publish is
# ``to_dict``/``from_dict``, and a constant here would be a second authority
# on a shape they own.


def is_foreign_model(stored: str | None, mine: str | None) -> bool:
    """Whether a vector recorded as *stored* was written by another model.

    The whole rule, in one place. It was in five, and two of the copies had
    already drifted --- both of them into false alarms, which is the
    direction that costs a caller a corpus re-embed.

    Two sides, and two ways to have nothing to say:

    * **An unnamed vector is unknown, not stale.** Vectors written before
      this key existed, and every vector from the ``embedding_fn`` lane,
      carry no name. Calling those foreign re-embeds a whole corpus on the
      first sweep after an upgrade.
    * **An unnamed reader accuses nobody.** An embedder publishing no
      ``model_id`` has one side of a comparison, not a disagreement with
      every row in the store. The registry states the same rule for the
      same case one module over: *"refusing would make a legitimate
      embedder unusable with a legitimate document"*.

    **Empty counts as unnamed on both sides.** A name is written from an
    embedder's ``model_id``, so ``""`` is what an embedder publishing
    nothing puts there; reporting it as a foreign model hands a caller an
    identity they cannot look up or re-embed against.

    Args:
        stored: What the vector recorded, or ``None`` where it recorded
            nothing.
        mine: The identity of the embedder asking, or ``None`` where it
            publishes none.

    Returns:
        True only where both sides name a model and the names differ.
    """
    return bool(stored) and bool(mine) and stored != mine


def row_model_name(metadata: dict[str, Any] | None) -> str | None:
    """The model a **stored row** says wrote it.

    The row container's reader: the flat :data:`MODEL_NAME_KEY`, and only
    that. The nested ``{"model": {"name": ...}}`` shape belongs to the
    sidecar container and is read by :func:`sidecar_model_name`; in a row,
    that spelling is caller data --- see the note above this function.

    **Three names for one thing, and they are one thing.** The key is
    spelled ``model_name``, what a writer puts in it is an embedder's
    ``model_id``, and the two results that carry it onward call it
    ``mismatched_model_ids``.

    Args:
        metadata: One stored row's metadata, as the store returned it.

    Returns:
        The name, or ``None`` where the row does not name one --- which
        includes naming it empty, the distinction :func:`is_foreign_model`
        explains.
    """
    if not metadata:
        return None
    name = metadata.get(MODEL_NAME_KEY)
    return str(name) if name else None


def foreign_model_names(
    metadatas: Iterable[dict[str, Any] | None],
    mine: str | None,
) -> list[str]:
    """Which models other than *mine* wrote these rows, distinct and sorted.

    What both store-row readers want, whole: ``SemanticIndex`` over one
    search's hits, ``DedupChecker`` over one check's candidates. What each
    does with the answer stays at each site and differs for stated reasons
    --- one warns once per *index* and accumulates across searches, the
    other warns once per *check*.

    **Distinct and sorted**, because the question is *which models*: a
    caller deciding what to re-embed wants the set, not one entry per row,
    and a stable order is what makes the answer assertable.

    Args:
        metadatas: One stored row's metadata each, in any order. ``None``
            entries are rows that carried none.
        mine: The identity of the embedder asking. ``None`` or empty
            answers ``[]`` --- see :func:`is_foreign_model`.

    Returns:
        Every foreign name found, once each, sorted.
    """
    found: set[str] = set()
    for metadata in metadatas:
        name = row_model_name(metadata)
        if name is not None and is_foreign_model(name, mine):
            found.add(name)
    return sorted(found)


def _sidecar_model_value(metadata: dict[str, Any] | None, nested: str, flat: str) -> Any:
    """Whichever of a sidecar's two shapes carries this value.

    **Nested wins where both are present**, because the nested one is what
    the declared shape writes. ``VectorField`` puts its constructor's
    ``model_name`` there while a caller's own ``metadata`` dict is merged in
    underneath, and ``VectorField.from_dict`` reads the nested one back.

    That precedence is the other reason a row is not read this way: a row
    written by ``bulk_embed_and_store`` carries the flat key from the
    embedder that just wrote it, over whatever a caller's metadata brought
    with it --- the opposite order, for the same reason. One function
    cannot hold both.

    The ``isinstance`` guards cross a persistence trust boundary, as
    :func:`stored_assembly` does: these values come back from whatever store
    wrote them and are not guaranteed to be the shapes that were written.
    """
    if not isinstance(metadata, dict):
        return None
    model = metadata.get("model")
    if isinstance(model, dict):
        value = model.get(nested)
        if value is not None:
            return value
    return metadata.get(flat)


def sidecar_model_name(metadata: dict[str, Any] | None) -> str | None:
    """The model a ``{field}_metadata`` **sidecar** says wrote its vector.

    Lives here rather than beside one of its readers for the reason
    :data:`MODEL_NAME_KEY` does: a container's reading rule belongs with the
    key, or it gets re-derived at the next site.

    **Both shapes, which is this container's tolerance and not a licence for
    the row reader to copy it.** ``VectorMetadata.to_dict`` nests the name as
    ``{"model": {"name": ...}}`` and that is what ``IncrementalVectorizer``
    writes; a hand-built sidecar may carry it flat. Reading only one shape is
    what made the version check compare against something nothing wrote.

    Args:
        metadata: The sidecar, or a ``VectorField``'s own metadata.

    Returns:
        The name, or ``None`` where the sidecar does not name one ---
        empty included, matching :func:`row_model_name` so that one rule
        answers both containers.
    """
    value = _sidecar_model_value(metadata, "name", MODEL_NAME_KEY)
    return str(value) if value else None


def sidecar_model_version(metadata: dict[str, Any] | None) -> str | None:
    """The model *version* a ``{field}_metadata`` sidecar records.

    The sibling of :func:`sidecar_model_name`, accepting the same two shapes
    for the same reason, and differing from it in one stated place:
    **absence is reported as absence, not folded into it.**

    The two keys are compared differently by their caller and have to be.
    An unnamed vector is passed over, because a corpus written before names
    were recorded must not all read as stale; an unversioned one is *not*,
    because a synchronizer tracking versions has nothing else to go on. So
    this returns ``""`` where a sidecar recorded an empty version rather
    than flattening it to ``None``, leaving that caller's distinction
    intact.

    Args:
        metadata: The sidecar, or a ``VectorField``'s own metadata.

    Returns:
        The version as written, or ``None`` where the sidecar records none.
    """
    value = _sidecar_model_value(metadata, "version", MODEL_VERSION_KEY)
    return str(value) if value is not None else None
