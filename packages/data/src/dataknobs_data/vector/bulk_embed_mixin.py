"""Mixin providing default bulk_embed_and_store implementation.

There are two of them, sync and async, and they were ~100-line near-copies
differing only in their ``await``s. That is what let every async backend mix in
the **sync** one for as long as it did: at the import site the two are
indistinguishable, and the wrong one raises nothing -- it returns a list of
un-awaited coroutines and stores no records at all.

So everything that is not the awaiting lives in the module-level helpers below,
and each mixin is the loop that drives them. What differs between the two is
now visible as the whole of what differs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..fields import VectorField
from .content import (
    DEFAULT_FIELD_SEPARATOR,
    assemble_source_text,
    compute_content_hash,
    content_hash_metadata,
)
from .embedding import embed_texts, require_embedding_source
from .types import BatchVectors

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

    from ..records import Record
    from .embedding import TextEmbedder


def resolve_text_fields(text_field: str | list[str]) -> list[str]:
    """Normalize the ``text_field`` argument to the list both mixins assemble."""
    return [text_field] if isinstance(text_field, str) else list(text_field)


def assemble_batch_texts(
    batch: list[Record],
    text_fields: list[str],
    field_separator: str,
) -> list[str]:
    """Build one embedder input per record, the way the whole package does.

    These were two more copies of the assembly loop, hardcoded to a space where
    the rest of the package had made the separator configurable.
    """
    return [assemble_source_text(record, text_fields, field_separator) for record in batch]


def pair_records_with_vectors(
    batch: list[Record],
    texts: list[str],
    embeddings: Any,
) -> Iterator[tuple[Record, Any, str]]:
    """Yield ``(record, vector, text)`` for each record an embedding covers.

    ``embedding_fn`` is only required to return something indexable and sized;
    a bare single embedding for a single text is tolerated, which is why the
    pairing is not simply ``zip``.
    """
    sized = hasattr(embeddings, "__len__")
    indexable = hasattr(embeddings, "__getitem__")

    for index, record in enumerate(batch):
        covered = index < len(embeddings) if sized else index == 0
        if not covered:
            continue
        vector = embeddings[index] if indexable else embeddings
        yield record, vector, texts[index]


def attach_vector_field(
    record: Record,
    vector_field: str,
    vector: Any,
    text: str,
    text_fields: list[str],
    field_separator: str,
    model_name: str | None,
    model_version: str | None,
) -> None:
    """Put the embedding on the record, described well enough to be judged.

    Without a digest the field cannot be judged stale by anything, so a
    synchronizer sweeping the same corpus treats it as current forever.
    """
    # Join multiple source fields with a comma for the legacy scalar key, which
    # is how its only reader parses it.
    source_field_str = text_fields[0] if len(text_fields) == 1 else ",".join(text_fields)

    record.fields[vector_field] = VectorField(
        name=vector_field,
        value=vector,
        source_field=source_field_str,
        model_name=model_name,
        model_version=model_version,
        metadata=content_hash_metadata(
            text_fields,
            field_separator,
            compute_content_hash(text),
        ),
    )


def track_vector_dimensions(store: Any, record: Record) -> None:
    """Let a backend that tracks vector dimensions see the new field."""
    if hasattr(store, "_has_vector_fields") and hasattr(store, "_update_vector_dimensions"):
        if store._has_vector_fields(record):
            store._update_vector_dimensions(record)


def iter_batches(records: list[Record], batch_size: int) -> Iterator[list[Record]]:
    """Slice the input into the batches handed to ``embedding_fn`` at once."""
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


class BulkEmbedMixin:
    """Mixin providing default implementation of bulk_embed_and_store.

    This mixin can be used by any **sync** database backend to provide a
    standard implementation of bulk embedding and storage without circular
    dependencies. Async backends want :class:`AsyncBulkEmbedMixin`.

    It takes no ``embedder``, and that is not an omission. A
    :class:`~dataknobs_data.vector.TextEmbedder` is async by declaration and
    this method cannot await one; giving the protocol a synchronous twin so
    that it could would put back the second shape the protocol exists to
    remove. A sync caller reaches an embedder from here by wrapping it once::

        with SyncTextEmbedder(embedder) as sync:
            store.bulk_embed_and_store(records, "body", embedding_fn=sync.embed)

    ``sync.embed`` already satisfies the ``embedding_fn`` parameter below, so
    the sync lane needs no new parameter to reach the seam.
    """

    def bulk_embed_and_store(
        self,
        records: list[Record],
        text_field: str | list[str],
        vector_field: str = "embedding",
        embedding_fn: Callable[[list[str]], BatchVectors] | None = None,
        batch_size: int = 100,
        model_name: str | None = None,
        model_version: str | None = None,
        field_separator: str = DEFAULT_FIELD_SEPARATOR,
    ) -> list[str]:
        """Embed text fields and store vectors with records.

        Args:
            records: Records to process
            text_field: Field name(s) containing text to embed
            vector_field: Field name to store vectors in
            embedding_fn: Function to generate embeddings. A
                ``SyncTextEmbedder``'s ``embed`` satisfies this --- see the
                class docstring for why there is no ``embedder`` parameter.
            batch_size: Number of records to process at once
            model_name: Name of the embedding model
            model_version: Version of the embedding model
            field_separator: What to join multiple text fields on. Was
                hardcoded to a space, which is the value it still defaults to.

        Returns:
            List of record IDs that were processed

        Raises:
            ValueError: If embedding_fn is not provided
        """
        if not embedding_fn:
            raise ValueError("embedding_fn is required for bulk_embed_and_store")

        text_fields = resolve_text_fields(text_field)
        processed_ids = []

        for batch in iter_batches(records, batch_size):
            texts = assemble_batch_texts(batch, text_fields, field_separator)
            if not texts:
                continue

            embeddings = embedding_fn(texts)

            for record, vector, text in pair_records_with_vectors(batch, texts, embeddings):
                attach_vector_field(
                    record,
                    vector_field,
                    vector,
                    text,
                    text_fields,
                    field_separator,
                    model_name,
                    model_version,
                )
                track_vector_dimensions(self, record)

                # Assumes self has create, update and exists (Database interface).
                if record.id and self.exists(record.id):  # type: ignore[attr-defined]
                    self.update(record.id, record)  # type: ignore[attr-defined]
                    processed_ids.append(record.id)
                else:
                    processed_ids.append(self.create(record))  # type: ignore[attr-defined]

        return processed_ids


class AsyncBulkEmbedMixin:
    """Async mixin providing default implementation of bulk_embed_and_store.

    Mixed into every async backend that offers the method. It was mixed into
    none of them for as long as it existed, which is a failure mode worth
    naming: the sync sibling they inherited instead is not a coroutine
    function, so its ``self.exists`` / ``self.update`` / ``self.create`` calls
    produced coroutines that were never awaited. A coroutine object is truthy,
    so the ``exists`` branch was taken unconditionally and nothing was ever
    written -- with no exception raised anywhere along the way.
    """

    async def bulk_embed_and_store(
        self,
        records: list[Record],
        text_field: str | list[str],
        vector_field: str = "embedding",
        embedding_fn: Callable[[list[str]], BatchVectors | Awaitable[BatchVectors]] | None = None,
        batch_size: int = 100,
        model_name: str | None = None,
        model_version: str | None = None,
        field_separator: str = DEFAULT_FIELD_SEPARATOR,
        *,
        embedder: TextEmbedder | None = None,
    ) -> list[str]:
        """Embed text fields and store vectors with records.

        Args:
            records: Records to process
            text_field: Field name(s) containing text to embed
            vector_field: Field name to store vectors in
            embedding_fn: Function to generate embeddings --- sync or async,
                returning an array or a list of vectors. All four combinations
                are handled, which is what ``embed_texts`` consolidated. Still
                accepted; prefer *embedder*.
            batch_size: Number of records to process at once
            model_name: Name of the embedding model. Defaults to the
                *embedder*'s own ``model_id`` when one is given, which is the
                point of the seam: the staleness key written beside a vector
                comes from the thing that produced it rather than from a
                parameter the caller keeps in step by hand.
            model_version: Version of the embedding model
            field_separator: What to join multiple text fields on. Was
                hardcoded to a space, which is the value it still defaults to.
            embedder: A :class:`~dataknobs_data.vector.TextEmbedder` --- async
                by declaration, so this lane classifies nothing when it is
                used.

        Returns:
            List of record IDs that were processed

        Raises:
            ValueError: Neither *embedding_fn* nor *embedder* was given, or
                both were.
        """
        # Checked before the loop, not inside it: an empty `records` never
        # reaches `embed_texts`, and "you gave me no embedder" must still be
        # an error there rather than a silently empty result.
        require_embedding_source(embedder, embedding_fn)

        text_fields = resolve_text_fields(text_field)
        if model_name is None and embedder is not None:
            model_name = embedder.model_id
        processed_ids = []

        for batch in iter_batches(records, batch_size):
            texts = assemble_batch_texts(batch, text_fields, field_separator)
            if not texts:
                continue

            embeddings = await embed_texts(texts, embedder=embedder, embedding_fn=embedding_fn)

            for record, vector, text in pair_records_with_vectors(batch, texts, embeddings):
                attach_vector_field(
                    record,
                    vector_field,
                    vector,
                    text,
                    text_fields,
                    field_separator,
                    model_name,
                    model_version,
                )
                track_vector_dimensions(self, record)

                # Assumes self has async create, update and exists
                # (AsyncDatabase interface).
                if record.id and await self.exists(record.id):  # type: ignore[attr-defined]
                    await self.update(record.id, record)  # type: ignore[attr-defined]
                    processed_ids.append(record.id)
                else:
                    processed_ids.append(await self.create(record))  # type: ignore[attr-defined]

        return processed_ids
