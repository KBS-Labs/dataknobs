"""Synchronization tools for keeping vectors up to date with text changes."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from dataknobs_common.callbacks import run_callback

from ..fields import VectorField
from ..records import Record
from .embedding import TextEmbedder, default_model_name, embed_text, require_embedding_source
from .content import (
    CONTENT_HASH_KEY,
    DEFAULT_FIELD_SEPARATOR,
    FIELD_SEPARATOR_KEY,
    MODEL_NAME_KEY,
    SOURCE_FIELDS_KEY,
    assemble_source_text,
    compute_content_hash,
    content_hash_metadata,
    current_content_hash,
    describes_its_assembly,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Collection, Coroutine

    from ..database import AsyncDatabase

logger = logging.getLogger(__name__)


def _stored_model_version(metadata: dict[str, Any]) -> str | None:
    """Read a model version out of a ``{field}_metadata`` sidecar.

    ``VectorMetadata.to_dict`` nests it as ``{"model": {"version": ...}}``,
    which is the shape ``IncrementalVectorizer`` writes. The flat
    ``model_version`` key was the only one read here and nothing writes it, so
    every record vectorized that way reported a version mismatch and was
    re-embedded on every sweep. Both shapes are accepted; the nested one is
    the one that exists.
    """
    model = metadata.get("model")
    if isinstance(model, dict):
        version = model.get("version")
        if version is not None:
            return str(version)
    version = metadata.get("model_version")
    return str(version) if version is not None else None


def _stored_model_name(metadata: dict[str, Any]) -> str | None:
    """Read a model name out of a ``{field}_metadata`` sidecar.

    The sibling of :func:`_stored_model_version`, accepting the same two
    shapes for the same reason: ``VectorMetadata.to_dict`` nests the name as
    ``{"model": {"name": ...}}`` and a hand-built sidecar may carry it flat.
    Reading only one shape is what made the version check compare against
    something nothing wrote.
    """
    model = metadata.get("model")
    if isinstance(model, dict):
        name = model.get("name")
        if name is not None:
            return str(name)
    name = metadata.get(MODEL_NAME_KEY)
    return str(name) if name is not None else None


@dataclass
class SyncConfig:
    """Configuration for vector synchronization."""

    auto_embed_on_create: bool = True
    auto_update_on_text_change: bool = True
    batch_size: int = 100
    track_model_version: bool = True
    # The identity an *embedder* supplies. `TextEmbedder` carries a `model_id`
    # and no version, so `bulk_embed_and_store(embedder=...)` defaults
    # `model_name` from it and leaves `model_version` unset --- which meant the
    # key the seam writes was not the key this class compared, and a model swap
    # read as current forever. Compared only when both sides carry a name, so a
    # corpus that never recorded one is unaffected.
    track_model_name: bool = True
    embedding_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.embedding_timeout <= 0:
            raise ValueError(f"Embedding timeout must be positive, got {self.embedding_timeout}")
        if self.max_retries < 0:
            raise ValueError(f"Max retries cannot be negative, got {self.max_retries}")


@dataclass
class SyncStatus:
    """Status of a synchronization operation."""

    total_records: int = 0
    processed_records: int = 0
    updated_records: int = 0
    failed_records: int = 0
    skipped_records: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the sync operation."""
        if self.processed_records == 0:
            return 0.0
        return (self.processed_records - self.failed_records) / self.processed_records

    @property
    def duration(self) -> float | None:
        """Calculate the duration of the sync operation in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "updated_records": self.updated_records,
            "failed_records": self.failed_records,
            "skipped_records": self.skipped_records,
            "success_rate": self.success_rate,
            "duration": self.duration,
            "errors": self.errors,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


class VectorTextSynchronizer:
    """Synchronizes vector embeddings with their source text fields."""

    def __init__(
        self,
        database: AsyncDatabase,
        embedding_fn: Callable[[str], np.ndarray]
        | Callable[[str], Coroutine[Any, Any, np.ndarray]]
        | None = None,
        text_fields: list[str] | str | None = None,
        vector_field: str = "embedding",
        field_separator: str = " ",
        auto_sync: bool = True,
        batch_size: int = 100,
        model_name: str | None = None,
        model_version: str | None = None,
        config: SyncConfig | None = None,
        *,
        embedder: TextEmbedder | None = None,
    ):
        """Initialize the synchronizer with simplified API.

        Args:
            database: The database to synchronize
            embedding_fn: Function to generate embeddings from text. Optional
                since *embedder* arrived; exactly one of the two is required.
            text_fields: Fields to concatenate for embedding (if None, uses all text fields)
            vector_field: Name of the vector field to store embeddings
            field_separator: Separator for concatenating text fields
            auto_sync: Whether to auto-sync on create/update
            batch_size: Batch size for bulk operations
            model_name: Name of the embedding model. Defaults to
                ``embedder.model_id`` when an *embedder* is given, which is
                what makes this class both the writer and the reader of one
                key --- see :meth:`_has_current_vector`.
            model_version: Version of the embedding model
            config: Advanced configuration object (overrides other params)
            embedder: A :class:`~dataknobs_data.vector.TextEmbedder`. Async by
                declaration, so nothing about it has to be classified before
                it is called.

        Raises:
            ValueError: Neither *embedding_fn* nor *embedder* was given, or
                both were. Checked here rather than at the first embed, so a
                misconfigured synchronizer fails where it is built rather than
                part-way through a sweep.
        """
        require_embedding_source(embedder, embedding_fn)

        self.database = database
        self.embedder = embedder
        self.embedding_fn = embedding_fn
        self.embedding_function = embedding_fn  # Alias for compatibility

        # Handle text_fields
        if isinstance(text_fields, str):
            text_fields = [text_fields]
        self.text_fields = text_fields or []

        self.vector_field = vector_field
        self.field_separator = field_separator
        self.auto_sync = auto_sync
        self.batch_size = batch_size
        # The embedder names itself, so a caller passing one does not also
        # have to keep `model_name` in step by hand --- which is the class of
        # error `model_id` exists to close, and this class is where the key is
        # read back. An explicit `model_name=` still wins: a caller who said
        # what they meant is not overridden.
        self.model_name = default_model_name(model_name, embedder)
        self.model_version = model_version

        # Use config if provided, otherwise create from params
        if config:
            self.config = config
        else:
            self.config = SyncConfig(
                auto_embed_on_create=auto_sync,
                auto_update_on_text_change=auto_sync,
                batch_size=batch_size,
            )
        self.config.validate()

        # Track vector fields and their source fields
        self._vector_fields: dict[str, dict[str, Any]] = {}
        self._source_fields: dict[str, list[str]] = defaultdict(list)
        self._initialize_field_mappings()

    def _initialize_field_mappings(self) -> None:
        """Register every vector field this synchronizer maintains.

        Two sources declare them, and both are registered here: the database
        schema, and the ``text_fields=`` argument of the simplified API. Only
        the schema was ever swept, which is why ``sync_on_update`` did nothing
        on the simplified path — the mapping it consults to learn which vector
        fields a source field feeds was empty, so it returned before doing any
        work. Registering both is also what lets ``sync_record`` maintain every
        field in one loop rather than one loop per source of truth.
        """
        # Use schema if available
        for field_name, field_schema in self.database.schema.fields.items():
            if field_schema.is_vector_field():
                source = field_schema.get_source_field()
                self._vector_fields[field_name] = {
                    "dimensions": field_schema.get_dimensions() or 384,
                    "source_field": source,
                    "source_fields": [source] if source else [],
                    "field_separator": DEFAULT_FIELD_SEPARATOR,
                }
                if source:
                    self._source_fields[source].append(field_name)

        # The simplified API names its own source fields and separator, which
        # override whatever the schema said about the same vector field.
        if self.text_fields:
            existing = self._vector_fields.get(self.vector_field, {})
            self._vector_fields[self.vector_field] = {
                **existing,
                "source_field": self.text_fields[0] if len(self.text_fields) == 1 else None,
                "source_fields": list(self.text_fields),
                "field_separator": self.field_separator,
            }
            # The override replaces the schema's account of this vector field,
            # so a source the schema named and `text_fields` does not no longer
            # feeds it. A stale entry left here makes `sync_on_update` re-embed
            # on an edit to a field the vector is not derived from -- and the
            # re-embed produces byte-identical text, so nothing downstream can
            # notice the work was pointless.
            for source, vector_fields in self._source_fields.items():
                if source not in self.text_fields and self.vector_field in vector_fields:
                    vector_fields.remove(self.vector_field)

            for source in self.text_fields:
                if self.vector_field not in self._source_fields[source]:
                    self._source_fields[source].append(self.vector_field)

    def _compute_content_hash(self, content: str) -> str:
        """Compute a hash of the content for change detection."""
        return compute_content_hash(content)

    def _has_current_vector(self, record: Record, vector_field: str) -> bool:
        """Check if a record has a current vector for the given field.

        Args:
            record: The record to check
            vector_field: Name of the vector field

        Returns:
            True if the vector is current, False otherwise
        """
        # Check if field exists
        field_obj = record.fields.get(vector_field)
        if not field_obj:
            return False

        # Get the vector value
        vector_value = None
        if isinstance(field_obj, VectorField):
            vector_value = field_obj.value
            if vector_value is None:
                return False

            # For VectorField, check model version if tracking is enabled
            if self.config.track_model_version and self.model_version:
                stored_version = field_obj.model_version
                if stored_version != self.model_version:
                    return False

            # `model_name` is the key the embedder seam actually writes. A
            # stored `None` is not a mismatch: it means the vector predates
            # anything recording a name, and calling every such vector stale
            # would re-embed a whole corpus on upgrade for no new information.
            if self.config.track_model_name and self.model_name:
                stored_name = field_obj.model_name
                if stored_name is not None and stored_name != self.model_name:
                    return False
        else:
            # Plain value (list or array)
            vector_value = field_obj.value
            if vector_value is None:
                return False
            if not isinstance(vector_value, (list, np.ndarray)):
                return False

            # For plain values, check metadata and content hash separately
            if self.config.track_model_version and self.model_version:
                metadata_field = f"{vector_field}_metadata"
                metadata = record.get_value(metadata_field)
                if not metadata or not isinstance(metadata, dict):
                    return False
                if _stored_model_version(metadata) != self.model_version:
                    return False

            # The same clause as the `VectorField` lane above, and it has to be
            # spelled differently to mean the same thing: there the name is an
            # attribute, here it is a sidecar record. An *absent* sidecar is
            # therefore the plain lane's spelling of "recorded no name", so it
            # cannot be a mismatch on its own --- unlike the version check
            # directly above, which does treat it as one.
            if self.config.track_model_name and self.model_name:
                metadata = record.get_value(f"{vector_field}_metadata")
                stored_name = _stored_model_name(metadata) if isinstance(metadata, dict) else None
                if stored_name is not None and stored_name != self.model_name:
                    return False

        # Compare the digest this class stored against the text the record
        # would produce now. The digest was previously written and never read:
        # a VectorField was treated as immutable once created, so an edited
        # source field left a stale vector in place and reported it current.
        #
        # Where the description lives is the whole of what the two lanes
        # differ by -- a `VectorField` keeps it on the field, a plain value in
        # a sidecar record field -- so only that much is asked here and the
        # rule itself is asked once, below. Written per lane, the digest check
        # reached only the `VectorField` one, which meant the same corpus and
        # the same edit came out differently depending on which class had
        # embedded it.
        description: dict[str, Any] | None
        if isinstance(field_obj, VectorField):
            description = field_obj.metadata
        else:
            sidecar = record.get_value(f"{vector_field}_metadata")
            description = sidecar if isinstance(sidecar, dict) else None

        return self._digest_is_current(record, vector_field, description)

    def _digest_is_current(
        self,
        record: Record,
        vector_field: str,
        description: dict[str, Any] | None,
    ) -> bool:
        """Whether a stored digest still matches the text the record holds now.

        Args:
            record: The record to reassemble the source text from.
            vector_field: The vector field being judged.
            description: Whatever the storage lane recorded beside the vector,
                or ``None`` if it recorded nothing.

        Returns:
            True if the vector is still current by digest.
        """
        stored_hash = (description or {}).get(CONTENT_HASH_KEY)
        if stored_hash is None:
            # Nothing to compare against. Hand-built fields and records
            # written before this class stored a digest are current, which
            # is what they were before the comparison existed — the new
            # behaviour is confined to fields this class can judge.
            return True

        # This class's own configuration, not the record's account of
        # itself. A synchronizer that deferred to the record could never
        # notice its own `text_fields` or `field_separator` changing: every
        # record would keep matching the assembly it was written under, so
        # the sweep meant to apply the new configuration would report
        # nothing to do and the change would never take effect. Reading the
        # record back is the *reader's* question -- see `.content`.
        field_info = self._vector_fields.get(vector_field) or {}
        current_hash = current_content_hash(
            record,
            field_info.get("source_fields") or [],
            field_info.get("field_separator", DEFAULT_FIELD_SEPARATOR),
        )
        return not (current_hash is not None and stored_hash != current_hash)

    def _needs_update(self, record: Record, vector_field: str) -> bool:
        """Check if a vector field needs to be updated.

        Args:
            record: The record to check
            vector_field: Name of the vector field

        Returns:
            True if the vector needs updating, False otherwise
        """
        return not self._has_current_vector(record, vector_field)

    def _describe_assembly(
        self,
        record: Record,
        vector_field: str,
        source_fields: list[str],
        separator: str,
    ) -> bool:
        """Record how a current vector's text is assembled, if it does not say.

        Called only for a field this synchronizer has just judged current under
        its own configuration, which is what makes the description true rather
        than a guess about the past: this class maintains the field, and will
        re-embed it under exactly this assembly from here on. A reader that
        repeats the description therefore gets the string this class would feed
        the embedder, which is the whole contract.

        Without this, the upgrade is one-way. A corpus digested on a
        non-default separator before descriptions existed reads as entirely
        outdated to a `ChangeTracker` -- which falls back to a space -- while
        this class correctly finds every record current and so never rewrites
        one. The two halves are each right and the corpus stays stuck.

        Returns:
            True if a description was added, meaning the record needs writing.
        """
        field_obj = record.fields.get(vector_field)
        if not isinstance(field_obj, VectorField):
            return False

        metadata = field_obj.metadata
        if not metadata or metadata.get(CONTENT_HASH_KEY) is None:
            # Nothing this class wrote, so nothing it can describe.
            return False
        if describes_its_assembly(metadata):
            return False

        metadata[SOURCE_FIELDS_KEY] = list(source_fields)
        metadata[FIELD_SEPARATOR_KEY] = separator
        logger.debug(
            "Described the assembly of %s on record %s: %s joined on %r",
            vector_field,
            record.id,
            source_fields,
            separator,
        )
        return True

    async def _embed_text(self, text: str) -> np.ndarray | None:
        """Generate embedding for text with error handling.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if failed
        """
        if not text:
            return None

        for attempt in range(self.config.max_retries):
            try:
                result = await embed_text(
                    text,
                    embedder=self.embedder,
                    embedding_fn=self.embedding_fn,
                    timeout=self.config.embedding_timeout,
                )

                if isinstance(result, np.ndarray):
                    return result
                elif isinstance(result, list):
                    return np.array(result)
                else:
                    logger.error(f"Embedding function returned unexpected type: {type(result)}")
                    return None

            except TimeoutError:
                logger.warning(f"Embedding timeout on attempt {attempt + 1}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
            except Exception as e:
                logger.error(f"Embedding error on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)

        return None

    async def sync_record(
        self,
        record_or_id: Record | str,
        force: bool = False,
        fields: Collection[str] | None = None,
    ) -> tuple[bool, list[str]]:
        """Synchronize vectors for a single record.

        The record is persisted whole, so the record handed in must be
        complete. Passing one built out of a partial update replaces the
        stored record with those fields alone.

        Args:
            record_or_id: The record or record ID to synchronize
            force: Force update even if vectors appear current
            fields: Restrict the work to these vector fields. ``None`` means
                every registered field. A caller that knows which vector
                fields a change can possibly have affected passes them here
                rather than forcing a re-embed of the ones it did not touch.

        Returns:
            Tuple of (success, list of re-embedded fields). ``success`` is
            ``False`` when the record could not be written, whatever was
            computed onto the copy the caller holds.
        """
        # Get record if ID provided
        record_id: str | None
        if isinstance(record_or_id, str):
            read_record = await self.database.read(record_or_id)
            if not read_record:
                return False, []
            record = read_record
            record_id = record_or_id
        else:
            record = record_or_id
            record_id = record.id

        updated_fields = []
        failed_fields = []
        described_fields = []

        # One loop over every registered vector field, whichever source of
        # truth declared it. The simplified `text_fields=` path used to be a
        # second branch above this one, and being separate is how it drifted:
        # it never consulted `_needs_update`, so it re-embedded unchanged
        # records on every sweep while the schema path skipped them.
        for vector_field_name, field_info in self._vector_fields.items():
            if fields is not None and vector_field_name not in fields:
                continue
            source_fields = field_info.get("source_fields") or []
            if not source_fields:
                continue
            separator = field_info.get("field_separator", DEFAULT_FIELD_SEPARATOR)

            if not (force or self._needs_update(record, vector_field_name)):
                # Current -- but possibly not self-describing. A record written
                # before the assembly was recorded carries a digest and no
                # account of how it was produced, so a reader falls back to its
                # own configuration and disagrees with this class on any
                # non-default separator. Permanently: nothing re-embeds a
                # record that is current, so the description would never be
                # written and the corpus could not heal. Writing it costs no
                # embedding.
                if self._describe_assembly(record, vector_field_name, source_fields, separator):
                    described_fields.append(vector_field_name)
                continue

            text = assemble_source_text(record, source_fields, separator)
            if not text:
                continue

            embedding = await self._embed_text(text)
            if embedding is None:
                failed_fields.append(vector_field_name)
                continue

            record.fields[vector_field_name] = VectorField(
                value=embedding,
                name=vector_field_name,
                source_field=source_fields[0] if len(source_fields) == 1 else None,
                model_name=self.model_name,
                model_version=self.model_version,
                # Digest the string that was just embedded, and describe how it
                # was assembled so a reader can reproduce it without being
                # configured the same way. See `.content`.
                metadata=content_hash_metadata(
                    source_fields, separator, compute_content_hash(text)
                ),
            )
            updated_fields.append(vector_field_name)

        # Save to database if anything on the record changed
        if updated_fields or described_fields:
            # Use storage_id if available, otherwise fall back to record.id
            update_id = record.storage_id if record.has_storage_id() else record_id
            if update_id is None:
                # No id to write under: the record was never stored. The
                # vectors are on the record the caller holds, so they are still
                # reported, but nothing was persisted and saying otherwise is
                # how this stayed invisible — `update(None, record)` does not
                # raise, it drops the write.
                logger.warning(
                    "Computed vector fields %s but did not persist them: record has no storage id",
                    updated_fields,
                )
                return False, updated_fields
            # `update` reports whether it found anything to write. Discarding
            # that is the same defect as passing `None` for the id, one layer
            # further in: a record whose id was never stored -- `Record(data=
            # {"id": "x"})` carries `id` without having been written -- is
            # reported synced against a database that has no such row.
            if not await self.database.update(update_id, record):
                logger.warning(
                    "Computed vector fields %s but did not persist them: "
                    "no record stored under id %s",
                    updated_fields,
                    update_id,
                )
                return False, updated_fields

        # Return success=False if there were failures and no successes
        success = len(failed_fields) == 0 or len(updated_fields) > 0
        return success, updated_fields

    async def sync_all(
        self,
        batch_size: int | None = None,
        force: bool = False,
        progress_callback: Callable[[int, int], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any]:
        """Synchronize all records in the database.

        Args:
            batch_size: Batch size for processing (uses self.batch_size if None)
            force: Force update even if vectors appear current
            progress_callback: Callback for progress updates (done, total)

        Returns:
            Dictionary with sync results
        """
        from ..query import Query

        batch_size = batch_size or self.batch_size

        # Get all records
        all_records = await self.database.search(Query())
        total = len(all_records)

        processed = 0
        updated = 0
        failed = 0

        # Process in batches
        for i in range(0, total, batch_size):
            batch = all_records[i : i + batch_size]

            for record in batch:
                success, fields = await self.sync_record(record, force=force)

                processed += 1
                if success and fields:
                    updated += 1
                elif not success:
                    failed += 1

                if progress_callback:
                    await run_callback(progress_callback, processed, total)

        return {
            "processed": processed,
            "updated": updated,
            "failed": failed,
            "total": total,
        }

    async def bulk_sync(
        self,
        records: list[Record] | None = None,
        force: bool = False,
        progress_callback: Callable[[SyncStatus], Awaitable[None] | None] | None = None,
    ) -> SyncStatus:
        """Synchronize vectors for multiple records in batches.

        Args:
            records: Records to sync (None for all records in database)
            force: Force update even if vectors appear current
            progress_callback: Callback for progress updates

        Returns:
            Synchronization status
        """
        status = SyncStatus(start_time=datetime.now(UTC))

        try:
            # Get records if not provided
            if records is None:
                records = await self.database.all()

            status.total_records = len(records)

            # Process in batches
            for i in range(0, len(records), self.config.batch_size):
                batch = records[i : i + self.config.batch_size]

                for record in batch:
                    try:
                        success, updated_fields = await self.sync_record(record, force)
                        status.processed_records += 1

                        # `success` first, matching `sync_all`. Counting a
                        # record as updated because fields were computed says
                        # nothing about whether they were stored -- which is
                        # exactly what `success` is now reporting.
                        if not success:
                            status.failed_records += 1
                        elif updated_fields:
                            # sync_record already updates the database
                            status.updated_records += 1
                        else:
                            status.skipped_records += 1

                    except Exception as e:
                        status.failed_records += 1
                        status.errors.append(
                            {
                                "record_id": record.id,
                                "error": str(e),
                            }
                        )
                        logger.error(f"Failed to sync record {record.id}: {e}")

                # Call progress callback
                if progress_callback:
                    await run_callback(progress_callback, status)

        finally:
            status.end_time = datetime.now(UTC)

        logger.info(
            f"Sync completed: {status.updated_records} updated, "
            f"{status.skipped_records} skipped, {status.failed_records} failed"
        )

        return status

    async def sync_on_update(
        self,
        record_id: str,
        old_data: dict[str, Any],
        new_data: dict[str, Any],
    ) -> bool:
        """Handle record updates and sync vectors if needed.

        ``new_data`` is what changed, not necessarily the whole record: an
        ``(old_data, new_data)`` signature invites a caller to pass only the
        fields it touched, and doing so must not cost it the rest of the
        record.

        Args:
            record_id: ID of the updated record
            old_data: Previous data
            new_data: New data, whole or partial

        Returns:
            True if vectors were re-embedded and stored, False otherwise
        """
        if not self.config.auto_update_on_text_change:
            return False

        # Check if any source fields changed
        fields_to_update = set()
        for source_field, vector_fields in self._source_fields.items():
            old_value = old_data.get(source_field)
            new_value = new_data.get(source_field)

            if old_value != new_value:
                fields_to_update.update(vector_fields)

        if not fields_to_update:
            return False

        # Apply the change to the *stored* record. `sync_record` persists the
        # record it is handed, whole, so handing it one built out of `new_data`
        # alone replaced the stored record with just the changed fields --
        # silently dropping every field the caller did not mention, and the
        # record's own metadata with them. This path only became reachable when
        # `text_fields=` started registering into `_source_fields`; before that
        # it returned above, which is why the loss had never been seen.
        stored = await self.database.read(record_id)
        if stored is not None:
            record = stored
            for key, value in new_data.items():
                record.set_value(key, value)
        else:
            record = Record(id=record_id, data=new_data)

        # Only the vector fields the changed sources actually feed. `force`
        # skips the staleness check for those, having just established
        # staleness directly; it is not a licence to re-embed the others.
        success, updated_fields = await self.sync_record(
            record, force=True, fields=fields_to_update
        )
        return success and bool(updated_fields)

    async def sync_on_create(self, record: Record) -> bool:
        """Handle record creation and sync vectors if needed.

        Args:
            record: The newly created record

        Returns:
            True if vectors were re-embedded and stored, False otherwise
        """
        if not self.config.auto_embed_on_create:
            return False

        # `sync_record` has already persisted whatever it computed, and already
        # reports whether that write landed. Writing again here was a second
        # identical round trip, and re-deriving the answer from `record.id`
        # duplicated -- less completely -- a judgement `success` already
        # carries.
        success, updated_fields = await self.sync_record(record)
        return success and bool(updated_fields)

    @classmethod
    def from_config(
        cls,
        database: AsyncDatabase,
        embedding_fn: Callable[[str], np.ndarray]
        | Callable[[str], Coroutine[Any, Any, np.ndarray]]
        | None = None,
        config: SyncConfig | None = None,
        text_fields: list[str] | None = None,
        vector_field: str = "embedding",
        model_name: str | None = None,
        model_version: str | None = None,
        *,
        embedder: TextEmbedder | None = None,
    ) -> VectorTextSynchronizer:
        """Create synchronizer from a config object for advanced use cases.

        Args:
            database: The database to synchronize
            embedding_fn: Function to generate embeddings from text. Optional
                since *embedder* arrived; exactly one of the two is required.
            config: Synchronization configuration. Optional so that
                ``embedder=`` may be passed by keyword without also restating
                a config; the constructor's default is used when omitted.
            text_fields: Text field names (optional)
            vector_field: Name of the vector field
            model_name: Name of the embedding model, defaulted from
                *embedder* when one is given
            model_version: Version of the embedding model
            embedder: A :class:`~dataknobs_data.vector.TextEmbedder`

        Returns:
            Configured VectorTextSynchronizer instance

        Raises:
            ValueError: Neither *embedding_fn* nor *embedder* was given, or
                both were.
        """
        resolved = config or SyncConfig()
        return cls(
            database=database,
            embedding_fn=embedding_fn,
            text_fields=text_fields,
            vector_field=vector_field,
            auto_sync=resolved.auto_embed_on_create,
            batch_size=resolved.batch_size,
            model_name=model_name,
            model_version=model_version,
            config=resolved,
            embedder=embedder,
        )
