# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Batch operations for DataKnobs-Pandas integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import pandas as pd

from dataknobs_common import BridgedOperation, SyncLoopBridge, bridged_operation
from dataknobs_common.callbacks import is_async_callable

from .converter import ConversionOptions, DataFrameConverter

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from contextlib import AbstractContextManager
    from dataknobs_data.database import AsyncDatabase, SyncDatabase
    from dataknobs_data.query import Query
    from dataknobs_data.records import Record


logger = logging.getLogger(__name__)

#: Loop-thread name for an operation's bridge. The bridge registers it, so
#: ``assert_no_leaked_bridge_threads`` watches this name too --- it is a
#: diagnostic label, not a way out of the leak guard.
BRIDGE_THREAD_NAME = "dk-sync-batch-ops"


@dataclass
class BatchConfig:
    """Configuration for batch operations."""

    chunk_size: int = 1000
    parallel: bool = False
    max_workers: int = 4
    progress_callback: Callable[[int, int], None] | None = None
    error_handling: str = "raise"  # "raise", "skip", "log"
    memory_efficient: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if self.error_handling not in ("raise", "skip", "log"):
            raise ValueError("error_handling must be one of: 'raise', 'skip', 'log'")


class ChunkedProcessor:
    """Process DataFrames in chunks for memory efficiency."""

    def __init__(self, chunk_size: int = 1000):
        """Initialize chunked processor.

        Args:
            chunk_size: Size of each chunk
        """
        self.chunk_size = chunk_size

    def process_dataframe(
        self,
        df: pd.DataFrame,
        processor: Callable[[pd.DataFrame], Any],
        combine: Callable[[list[Any]], Any] | None = None,
    ) -> Any:
        """Process DataFrame in chunks.

        Args:
            df: DataFrame to process
            processor: Function to process each chunk
            combine: Function to combine results

        Returns:
            Combined results or list of chunk results
        """
        results = []

        for chunk in self.iter_chunks(df):
            result = processor(chunk)
            results.append(result)

        if combine:
            return combine(results)
        return results

    def iter_chunks(self, df: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
        """Iterate over DataFrame in chunks.

        Args:
            df: DataFrame to chunk

        Yields:
            DataFrame chunks
        """
        for start_idx in range(0, len(df), self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, len(df))
            yield df.iloc[start_idx:end_idx]

    def read_csv_chunked(
        self, filepath: str, processor: Callable[[pd.DataFrame], Any], **read_kwargs: Any
    ) -> list[Any]:
        """Read CSV file in chunks and process.

        Args:
            filepath: Path to CSV file
            processor: Function to process each chunk
            **read_kwargs: Additional arguments for pd.read_csv

        Returns:
            List of processed results
        """
        results = []

        for chunk in pd.read_csv(filepath, chunksize=self.chunk_size, **read_kwargs):
            result = processor(chunk)
            results.append(result)

        return results


class BatchOperations:
    """Batch operations for DataKnobs databases using DataFrames.

    Fronts either flavour of database. A :class:`SyncDatabase` is called
    directly. An :class:`AsyncDatabase` is reached through a
    :class:`~dataknobs_common.sync_bridge.SyncLoopBridge` --- a private event
    loop on a daemon thread --- so every method here is callable from plain
    synchronous code *and* from inside a running event loop. Driving the
    coroutine on the caller's own loop instead raises ``RuntimeError`` in the
    second case, which is the case a synchronous helper over an async store
    exists to serve.

    "Callable from inside a running loop" means it does not deadlock. It still
    **blocks**: the calling thread waits for the whole operation, so every
    other task on the caller's loop is stalled for as long as the database
    takes. From async code, ``await`` the database directly --- this class is
    for the ``def`` sites that cannot. Pass ``timeout`` for an upper bound on
    a wait a synchronous caller has no other way to cancel; it bounds the
    operation, not each database round trip inside it.

    The loop is **operation-scoped**: one public call gets one loop, shared by
    every coroutine that call drives --- every chunk of a
    :meth:`bulk_insert_dataframe`, every row of its per-record fallback, and
    both halves of a :meth:`transform_and_save` --- and the thread ends with
    the call. So this class acquires no teardown obligation: there is nothing
    to ``close``, and a synchronous database never allocates a thread at all.

    .. important::

       **A backend that binds loop state to its connection needs a bridge you
       supply.** ``AsyncPostgresDatabase`` acquires an ``asyncpg`` pool in
       ``connect()``, and that pool belongs to the loop that acquired it. This
       class does not own the database, so it cannot own that loop: whichever
       loop *you* connected on is the one every later operation must use.
       Pass it as ``bridge``::

           with SyncLoopBridge() as bridge:
               bridge.run(db.connect())
               ops = BatchOperations(db, bridge=bridge)
               ops.bulk_insert_dataframe(df)

       A bridge given here belongs to the caller: it is shared with whatever
       else uses it, and nothing in this class closes it. Without one, each
       operation runs on a loop of its own and a pooled backend raises
       ``InterfaceError: cannot perform operation: another operation is in
       progress`` --- from synchronous code, with no running loop anywhere,
       because ``connect()``'s loop is already gone. Backends holding no
       loop-bound state (memory, file) are unaffected either way.
    """

    def __init__(
        self,
        database: AsyncDatabase | SyncDatabase,
        converter: DataFrameConverter | None = None,
        *,
        bridge: SyncLoopBridge | None = None,
        timeout: float | None = None,
    ):
        """Initialize batch operations.

        Args:
            database: Target database, of either flavour.
            converter: DataFrame converter.
            bridge: A bridge to run this object's coroutines on, for the whole
                of its life rather than one operation at a time. Required when
                the database holds state bound to the loop that connected it
                --- see the class docstring. It belongs to the caller: nothing
                here closes it. The default gives each operation a private
                bridge and ends it with the operation.
            timeout: Seconds to allow one *operation* --- one public call,
                however many database round trips it makes --- giving a
                synchronous caller an upper bound on a blocking wait it cannot
                otherwise cancel. A call that finds the deadline already past
                raises :class:`TimeoutError` without reaching the database.
                ``None`` (the default) waits as long as the database takes.
                Ignored for a synchronous database.
        """
        self.database = database
        self.converter = converter or DataFrameConverter()
        self.is_async = hasattr(database, "create") and is_async_callable(database.create)
        self._bridge = bridge
        self._timeout = timeout

    # -- reaching the database -------------------------------------------

    def _operation_loop(self) -> AbstractContextManager[BridgedOperation]:
        """Open the loop and the time budget this operation runs within.

        The scope is the *public entry point*, not the database call: the
        private workers take the operation as an argument, so a composite
        method drives both of its halves on one loop instead of stranding the
        first half's, and one ``timeout`` bounds the call the caller actually
        made rather than each database round trip inside it.

        ``needs_loop`` is what keeps a synchronous database from being charged
        a thread for a loop it will never reach; the operation still carries
        the deadline, because a synchronous caller can ask for a bound too.
        """
        return bridged_operation(
            bridge=self._bridge,
            timeout=self._timeout,
            thread_name=BRIDGE_THREAD_NAME,
            label="BatchOperations",
            needs_loop=self.is_async,
        )

    def _search(self, op: BridgedOperation, query: Query) -> list[Record]:
        """``search``, on whichever flavour of database this object fronts."""
        if self.is_async:
            return op.run(cast("AsyncDatabase", self.database).search(query))
        return cast("SyncDatabase", self.database).search(query)

    def _create(self, op: BridgedOperation, record: Record) -> str:
        """``create``, on whichever flavour of database this object fronts."""
        if self.is_async:
            return op.run(cast("AsyncDatabase", self.database).create(record))
        return cast("SyncDatabase", self.database).create(record)

    def _create_batch(self, op: BridgedOperation, records: list[Record]) -> list[str]:
        """``create_batch``, on whichever flavour of database this object fronts."""
        if self.is_async:
            return op.run(cast("AsyncDatabase", self.database).create_batch(records))
        return cast("SyncDatabase", self.database).create_batch(records)

    def _update(self, op: BridgedOperation, record_id: str, record: Record) -> bool:
        """``update``, on whichever flavour of database this object fronts."""
        if self.is_async:
            return op.run(cast("AsyncDatabase", self.database).update(record_id, record))
        return cast("SyncDatabase", self.database).update(record_id, record)

    def _update_batch(self, op: BridgedOperation, updates: list[tuple[str, Record]]) -> list[bool]:
        """``update_batch``, on whichever flavour of database this object fronts."""
        if self.is_async:
            return op.run(cast("AsyncDatabase", self.database).update_batch(updates))
        return cast("SyncDatabase", self.database).update_batch(updates)

    def bulk_insert_dataframe(
        self,
        df: pd.DataFrame,
        config: BatchConfig | None = None,
        conversion_options: ConversionOptions | None = None,
    ) -> dict[str, Any]:
        """Bulk insert DataFrame rows into database.

        Every chunk --- and, on the per-record fallback path, every row ---
        runs on the one loop this operation holds.

        Args:
            df: DataFrame to insert
            config: Batch configuration
            conversion_options: Options for DataFrame conversion

        Returns:
            Insert statistics
        """
        with self._operation_loop() as op:
            return self._bulk_insert_dataframe(op, df, config, conversion_options)

    def _bulk_insert_dataframe(
        self,
        op: BridgedOperation,
        df: pd.DataFrame,
        config: BatchConfig | None = None,
        conversion_options: ConversionOptions | None = None,
    ) -> dict[str, Any]:
        """:meth:`bulk_insert_dataframe`, on a loop the caller already holds."""
        config = config or BatchConfig()
        conversion_options = conversion_options or ConversionOptions()
        # These are now guaranteed to be non-None
        assert config is not None
        assert conversion_options is not None

        stats: dict[str, Any] = {"total_rows": len(df), "inserted": 0, "failed": 0, "errors": []}

        # Process in chunks if memory efficient mode
        if config.memory_efficient and len(df) > config.chunk_size:
            processor = ChunkedProcessor(config.chunk_size)
            # Create local references that are guaranteed non-None
            final_config = config
            final_conversion_options = conversion_options

            def process_chunk(chunk_df: pd.DataFrame) -> dict[str, int]:
                return self._insert_chunk(op, chunk_df, final_config, final_conversion_options)

            chunk_results = processor.process_dataframe(df, process_chunk)

            # Aggregate results
            for result in chunk_results:
                stats["inserted"] += result["inserted"]
                stats["failed"] += result["failed"]
                if "errors" in result:
                    stats["errors"].extend(result["errors"])
        else:
            # Process entire DataFrame at once
            stats = self._insert_chunk(op, df, config, conversion_options)

        return stats

    def query_as_dataframe(
        self, query: Query, conversion_options: ConversionOptions | None = None
    ) -> pd.DataFrame:
        """Execute query and return results as DataFrame.

        Args:
            query: Query to execute
            conversion_options: Options for conversion

        Returns:
            Query results as DataFrame
        """
        with self._operation_loop() as op:
            return self._query_as_dataframe(op, query, conversion_options)

    def _query_as_dataframe(
        self,
        op: BridgedOperation,
        query: Query,
        conversion_options: ConversionOptions | None = None,
    ) -> pd.DataFrame:
        """:meth:`query_as_dataframe`, on a loop the caller already holds."""
        conversion_options = conversion_options or ConversionOptions()

        records = self._search(op, query)

        # Convert to DataFrame
        return self.converter.records_to_dataframe(records, conversion_options)

    def update_from_dataframe(
        self,
        df: pd.DataFrame,
        id_column: str | None,
        config: BatchConfig | None = None,
        conversion_options: ConversionOptions | None = None,
    ) -> dict[str, Any]:
        """Update records from DataFrame using ID column.

        Every chunk --- and, on the per-record fallback path, every row ---
        runs on the one loop this operation holds.

        Args:
            df: DataFrame with updates
            id_column: Column containing record IDs
            config: Batch configuration
            conversion_options: Conversion options

        Returns:
            Update statistics
        """
        with self._operation_loop() as op:
            return self._update_from_dataframe(op, df, id_column, config, conversion_options)

    def _update_from_dataframe(
        self,
        op: BridgedOperation,
        df: pd.DataFrame,
        id_column: str | None,
        config: BatchConfig | None = None,
        conversion_options: ConversionOptions | None = None,
    ) -> dict[str, Any]:
        """:meth:`update_from_dataframe`, on a loop the caller already holds."""
        config = config or BatchConfig()
        conversion_options = conversion_options or ConversionOptions()

        stats: dict[str, Any] = {
            "total_rows": len(df),
            "updated": 0,
            "failed": 0,
            "not_found": 0,
            "errors": [],
        }

        # Convert DataFrame to records
        records = self.converter.dataframe_to_records(df, conversion_options)

        # Prepare updates as (id, record) tuples
        updates = []
        if id_column is None:
            # Use index as ID source
            for idx, record in zip(df.index, records, strict=True):
                record_id = str(idx)
                updates.append((record_id, record))
        else:
            # Ensure ID column exists
            if id_column not in df.columns:
                raise ValueError(f"ID column '{id_column}' not found in DataFrame")
            # Use specified column as ID source
            for i, record in enumerate(records):
                record_id = str(df.iloc[i][id_column])
                updates.append((record_id, record))

        # Process updates in chunks
        for i in range(0, len(updates), config.chunk_size):
            chunk = updates[i : i + config.chunk_size]

            try:
                # Use batch update for better performance
                results = self._update_batch(op, chunk)

                # Count successes and failures
                for success in results:
                    if success:
                        stats["updated"] += 1
                    else:
                        stats["not_found"] += 1

            except Exception:
                # If batch fails, try individual updates
                if config.error_handling == "raise":
                    raise

                for record_id, record in chunk:
                    try:
                        success = self._update(op, record_id, record)

                        if success:
                            stats["updated"] += 1
                        else:
                            stats["not_found"] += 1

                    except Exception as e:
                        stats["failed"] += 1
                        if config.error_handling == "log":
                            logger.error(f"Failed to update record {record_id}: {e}")
                            stats["errors"].append(str(e))
                        # else "skip"

            # Progress callback
            if config.progress_callback:
                processed = stats["updated"] + stats["failed"] + stats["not_found"]
                config.progress_callback(processed, len(updates))

        return stats

    def aggregate(
        self,
        query: Query,
        aggregations: dict[str, str | Callable],
        group_by: list[str] | None = None,
    ) -> pd.DataFrame:
        """Perform aggregations on query results.

        Args:
            query: Query to execute
            aggregations: Dictionary of column: aggregation function
            group_by: Columns to group by

        Returns:
            Aggregated DataFrame
        """
        # Get data as DataFrame
        with self._operation_loop() as op:
            df = self._query_as_dataframe(op, query)

        if df.empty:
            return pd.DataFrame()

        # Perform aggregations
        if group_by:
            grouped = df.groupby(group_by)
            return grouped.agg(aggregations)
        else:
            # Single row with aggregations
            result = {}
            for col, agg_func in aggregations.items():
                if col in df.columns:
                    if isinstance(agg_func, str):
                        result[f"{col}_{agg_func}"] = df[col].agg(agg_func)
                    else:
                        result[f"{col}_agg"] = agg_func(df[col])
            return pd.DataFrame([result])

    def transform_and_save(
        self,
        query: Query,
        transformer: Callable[[pd.DataFrame], pd.DataFrame],
        config: BatchConfig | None = None,
    ) -> dict[str, Any]:
        """Query, transform with pandas, and save back.

        Args:
            query: Query to get records
            transformer: Function to transform DataFrame
            config: Batch configuration

        Returns:
            Operation statistics
        """
        config = config or BatchConfig()

        # One loop for the read and the write back: the halves are one
        # operation against one database, and giving the second half a loop of
        # its own is what strands a connection the first half bound.
        with self._operation_loop() as op:
            df = self._query_as_dataframe(op, query)

            if df.empty:
                return {"total_rows": 0, "transformed": 0}

            # Apply transformation
            transformed_df = transformer(df)

            # Save back if index preserved (has record IDs)
            if df.index.name == "record_id" and transformed_df.index.name == "record_id":
                return self._update_from_dataframe(
                    op,
                    transformed_df,
                    id_column=None,  # Use index
                    config=config,
                )
            # Insert as new records
            return self._bulk_insert_dataframe(op, transformed_df, config)

    def _insert_chunk(
        self,
        op: BridgedOperation,
        df: pd.DataFrame,
        config: BatchConfig,
        conversion_options: ConversionOptions,
    ) -> dict[str, Any]:
        """Insert a chunk of DataFrame rows, on the operation's loop.

        Both fallback paths below call the database once per row, which is why
        the loop is the *operation's* rather than each call's: a bridge per
        row would be a daemon thread per row.

        Args:
            op: The loop this operation's coroutines run on and the deadline
                they share; its bridge is ``None`` for a synchronous database,
                which reaches no loop.
            df: DataFrame chunk
            config: Batch configuration
            conversion_options: Conversion options

        Returns:
            Insert statistics for chunk
        """
        stats: dict[str, Any] = {"total_rows": len(df), "inserted": 0, "failed": 0, "errors": []}

        # Convert to records
        records = self.converter.dataframe_to_records(df, conversion_options)

        # Use batch creation for better performance with graceful fallback
        if hasattr(self.database, "create_batch"):
            try:
                ids = self._create_batch(op, records)
                stats["inserted"] = len(ids)

                # Progress callback for successful batch
                if config.progress_callback:
                    config.progress_callback(len(records), len(records))

            except Exception as batch_error:
                # Retrying row by row is what identifies *which* rows are bad,
                # which is why this path exists and why it runs whatever
                # `error_handling` says. What it must not do is turn the batch
                # failure into a success: when every row then writes fine ---
                # a batch size limit, a transient, a timed-out batch --- the
                # caller used to be told nothing had gone wrong at all, having
                # asked (by default) to be stopped. `batch_error_stands`
                # carries that question past the loop.
                batch_error_stands = True
                for i, record in enumerate(records):
                    try:
                        self._create(op, record)
                        stats["inserted"] += 1

                    except Exception as record_error:
                        # A row failed too, so the row's error is the specific
                        # one and the batch error adds nothing.
                        batch_error_stands = False
                        stats["failed"] += 1

                        # Handle error based on config
                        if config.error_handling == "raise":
                            raise
                        elif config.error_handling == "log":
                            logger.error(f"Failed to insert row {i}: {record_error}")
                            stats["errors"].append(str(record_error))
                        # else "skip" - just continue

                    # Progress callback for each record
                    if config.progress_callback:
                        config.progress_callback(i + 1, len(records))

                if batch_error_stands:
                    # Every row went in individually, so only the batch write
                    # failed --- and nothing above has reported it.
                    if config.error_handling == "raise":
                        raise
                    if config.error_handling == "log":
                        logger.error(
                            "Batch insert failed; all %d rows were written individually: %s",
                            len(records),
                            batch_error,
                        )
                        stats["errors"].append(str(batch_error))
                    # else "skip" - the caller asked not to hear about it
        else:
            # Fallback to individual inserts if create_batch not available
            for i, record in enumerate(records):
                try:
                    self._create(op, record)
                    stats["inserted"] += 1

                except Exception as e:
                    stats["failed"] += 1
                    if config.error_handling == "raise":
                        raise
                    elif config.error_handling == "log":
                        logger.error(f"Failed to insert row {i}: {e}")
                        stats["errors"].append(str(e))
                    # else "skip"

                # Progress callback
                if config.progress_callback:
                    config.progress_callback(i + 1, len(records))

        return stats

    def export_to_csv(
        self,
        query: Query,
        filepath: str,
        conversion_options: ConversionOptions | None = None,
        **to_csv_kwargs: Any,
    ) -> None:
        """Export query results to CSV file.

        Args:
            query: Query to execute
            filepath: Output file path
            conversion_options: Conversion options
            **to_csv_kwargs: Additional arguments for DataFrame.to_csv
        """
        with self._operation_loop() as op:
            df = self._query_as_dataframe(op, query, conversion_options)
        df.to_csv(filepath, **to_csv_kwargs)

    def export_to_parquet(
        self,
        query: Query,
        filepath: str,
        conversion_options: ConversionOptions | None = None,
        **to_parquet_kwargs: Any,
    ) -> None:
        """Export query results to Parquet file.

        Args:
            query: Query to execute
            filepath: Output file path
            conversion_options: Conversion options
            **to_parquet_kwargs: Additional arguments for DataFrame.to_parquet
        """
        with self._operation_loop() as op:
            df = self._query_as_dataframe(op, query, conversion_options)
        df.to_parquet(filepath, **to_parquet_kwargs)
