"""Mixin providing default bulk_embed_and_store implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ..fields import VectorField
from .content import (
    DEFAULT_FIELD_SEPARATOR,
    assemble_source_text,
    compute_content_hash,
    content_hash_metadata,
)

if TYPE_CHECKING:
    import numpy as np
    from collections.abc import Awaitable, Callable
    from ..records import Record


class BulkEmbedMixin:
    """Mixin providing default implementation of bulk_embed_and_store.

    This mixin can be used by any database backend to provide a standard
    implementation of bulk embedding and storage without circular dependencies.
    """

    def bulk_embed_and_store(
        self,
        records: list[Record],
        text_field: str | list[str],
        vector_field: str = "embedding",
        embedding_fn: Callable[[list[str]], np.ndarray] | None = None,
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
            embedding_fn: Function to generate embeddings
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

        # Process text fields
        if isinstance(text_field, str):
            text_fields = [text_field]
        else:
            text_fields = text_field

        processed_ids = []

        # Process in batches
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]

            # Extract text from records, the same way every other class in
            # this package extracts it. These were two more copies of the loop,
            # hardcoded to a space where the rest of the package had made the
            # separator configurable.
            texts = [assemble_source_text(record, text_fields, field_separator) for record in batch]

            # Generate embeddings
            if texts:
                embeddings = embedding_fn(texts)

                # Add vectors to records
                for j, record in enumerate(batch):
                    if j < len(embeddings) if hasattr(embeddings, "__len__") else j == 0:
                        # Get the embedding for this record
                        if hasattr(embeddings, "__getitem__"):
                            vector = embeddings[j]
                        else:
                            # Single embedding returned for single text
                            vector = embeddings

                        # Add or update vector field
                        # Join multiple source fields with comma for metadata
                        source_field_str = (
                            text_fields[0] if len(text_fields) == 1 else ",".join(text_fields)
                        )
                        record.fields[vector_field] = VectorField(
                            name=vector_field,
                            value=vector,
                            source_field=source_field_str,
                            model_name=model_name,
                            model_version=model_version,
                            # Without a digest the field cannot be judged
                            # stale by anything, so a synchronizer sweeping
                            # the same corpus treats it as current forever.
                            metadata=content_hash_metadata(
                                text_fields,
                                field_separator,
                                compute_content_hash(texts[j]),
                            ),
                        )

                        # Update vector dimensions tracking if available
                        if hasattr(self, "_has_vector_fields") and hasattr(
                            self, "_update_vector_dimensions"
                        ):
                            if self._has_vector_fields(record):
                                self._update_vector_dimensions(record)

                        # Create or update the record
                        # Assumes self has create, update, and exists methods (from Database interface)
                        if record.id and self.exists(record.id):  # type: ignore
                            self.update(record.id, record)  # type: ignore
                            processed_ids.append(record.id)
                        else:
                            record_id = self.create(record)  # type: ignore
                            processed_ids.append(record_id)

        return processed_ids


class AsyncBulkEmbedMixin:
    """Async mixin providing default implementation of bulk_embed_and_store.

    This mixin can be used by any async database backend to provide a standard
    implementation of bulk embedding and storage without circular dependencies.
    """

    async def bulk_embed_and_store(
        self,
        records: list[Record],
        text_field: str | list[str],
        vector_field: str = "embedding",
        embedding_fn: Callable[[list[str]], np.ndarray | Awaitable[np.ndarray]] | None = None,
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
            embedding_fn: Function to generate embeddings (can be sync or async)
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
        import inspect

        if not embedding_fn:
            raise ValueError("embedding_fn is required for bulk_embed_and_store")

        # Check if embedding_fn is async
        is_async_fn = inspect.iscoroutinefunction(embedding_fn)

        # Process text fields
        if isinstance(text_field, str):
            text_fields = [text_field]
        else:
            text_fields = text_field

        processed_ids = []

        # Process in batches
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]

            # Extract text from records, the same way every other class in
            # this package extracts it. These were two more copies of the loop,
            # hardcoded to a space where the rest of the package had made the
            # separator configurable.
            texts = [assemble_source_text(record, text_fields, field_separator) for record in batch]

            # Generate embeddings
            if texts:
                if is_async_fn:
                    embeddings = await cast("Awaitable[np.ndarray]", embedding_fn(texts))
                else:
                    embeddings = cast("np.ndarray", embedding_fn(texts))

                # Add vectors to records
                for j, record in enumerate(batch):
                    if j < len(embeddings) if hasattr(embeddings, "__len__") else j == 0:
                        # Get the embedding for this record
                        if hasattr(embeddings, "__getitem__"):
                            vector = embeddings[j]
                        else:
                            # Single embedding returned for single text
                            vector = embeddings

                        # Add or update vector field
                        # Join multiple source fields with comma for metadata
                        source_field_str = (
                            text_fields[0] if len(text_fields) == 1 else ",".join(text_fields)
                        )
                        record.fields[vector_field] = VectorField(
                            name=vector_field,
                            value=vector,
                            source_field=source_field_str,
                            model_name=model_name,
                            model_version=model_version,
                            # Without a digest the field cannot be judged
                            # stale by anything, so a synchronizer sweeping
                            # the same corpus treats it as current forever.
                            metadata=content_hash_metadata(
                                text_fields,
                                field_separator,
                                compute_content_hash(texts[j]),
                            ),
                        )

                        # Update vector dimensions tracking if available
                        if hasattr(self, "_has_vector_fields") and hasattr(
                            self, "_update_vector_dimensions"
                        ):
                            if self._has_vector_fields(record):
                                self._update_vector_dimensions(record)

                        # Create or update the record
                        # Assumes self has async create, update, and exists methods (from AsyncDatabase interface)
                        if record.id and await self.exists(record.id):  # type: ignore
                            await self.update(record.id, record)  # type: ignore
                            processed_ids.append(record.id)
                        else:
                            record_id = await self.create(record)  # type: ignore
                            processed_ids.append(record_id)

        return processed_ids
