"""Built-in database functions for FSM.

This module provides database-related functions that can be referenced
in FSM configurations, leveraging the dataknobs_data package.
"""

from collections.abc import Callable, Mapping
from typing import Any, Dict, List

from dataknobs_common import CapabilityNotSupportedError
from dataknobs_common.exceptions import ConfigurationError, ValidationError

from dataknobs_fsm.functions.base import ITransformFunction, TransformError
from dataknobs_fsm.functions.library.identity import (
    RecordIdentity,
    resolve_identity,
)


def _resources_from_context(context: Any) -> Dict[str, Any]:
    """Extract the injected resources mapping from a function context.

    The async engine passes a :class:`~dataknobs_fsm.functions.base.FunctionContext`
    whose ``resources`` dict is populated from the state's declared resource
    requirements. Tolerate a plain dict context (and ``None``) so these
    functions are callable outside the engine too.
    """
    if context is None:
        return {}
    if isinstance(context, dict):
        resources = context.get("resources", {})
    else:
        resources = getattr(context, "resources", {})
    return resources or {}


def _require_resource(resource_name: str, context: Any) -> Any:
    """Return the named resource from ``context.resources`` or raise.

    Resources are injected by the engine into ``FunctionContext.resources``
    from the state's ``resources`` declaration — not smuggled through the data
    payload. A missing resource is a wiring error (the state did not declare
    the resource, or no provider is registered for it).
    """
    resource = _resources_from_context(context).get(resource_name)
    if resource is None:
        raise TransformError(
            f"Database resource '{resource_name}' not found in "
            f"context.resources (is it declared in the state's 'resources'?)"
        )
    return resource


class DatabaseFetch(ITransformFunction):
    """Fetch data from a database using a query."""

    def __init__(
        self,
        resource_name: str,
        query: str,
        params: Dict[str, Any] | None = None,
        fetch_one: bool = False,
        as_dict: bool = True,
    ):
        """Initialize the database fetch function.
        
        Args:
            resource_name: Name of the database resource to use.
            query: SQL query to execute.
            params: Query parameters for parameterized queries.
            fetch_one: If True, fetch only one record.
            as_dict: If True, return records as dictionaries.
        """
        self.resource_name = resource_name
        self.query = query
        self.params = params or {}
        self.fetch_one = fetch_one
        self.as_dict = as_dict

    async def transform(
        self, data: Dict[str, Any], context: Any = None
    ) -> Dict[str, Any]:
        """Transform data by fetching from database.
        
        Args:
            data: Input data (can contain query parameters).
            
        Returns:
            Data with database query results.
        """
        # Resource is injected by the engine into context.resources.
        resource = _require_resource(self.resource_name, context)
        
        # Merge parameters
        query_params = {**self.params}
        
        # Allow dynamic parameters from input data
        for key, value in data.items():
            if key.startswith("param_"):
                param_name = key[6:]  # Remove "param_" prefix
                query_params[param_name] = value
        
        try:
            # Execute query
            result = await resource.execute_query(
                self.query,
                params=query_params,
                fetch_one=self.fetch_one,
                as_dict=self.as_dict,
            )
            
            # Return result
            if self.fetch_one:
                return {"record": result, **data}
            else:
                return {"records": result, **data}
        
        except Exception as e:
            raise TransformError(f"Database query failed: {e}") from e

    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        return f"Fetch data from {self.resource_name} using query: {self.query[:50]}..."


class DatabaseUpsert(ITransformFunction):
    """Upsert data into a database table."""

    def __init__(
        self,
        resource_name: str,
        table: str,
        key_columns: List[str] | None = None,
        value_columns: List[str] | None = None,
        on_conflict: str = "update",  # "update", "ignore", "error"
        *,
        id_fn: Callable[[Mapping[str, Any]], str | None] | None = None,
        identity: RecordIdentity | None = None,
    ):
        """Initialize the database upsert function.

        Args:
            resource_name: Name of the database resource to use.
            table: Table name to upsert into.
            key_columns: Columns that form the unique key. Sugar for a
                :class:`KeyColumnsIdentity`; also scopes the ``value_columns``
                projection.
            value_columns: Columns to update (if None, update all non-key columns).
            on_conflict: Action on conflict ("update", "ignore", "error").
            id_fn: Sugar for a :class:`CallableIdentity` deriving each row's id.
            identity: An explicit :class:`RecordIdentity`. Specify at most one of
                ``key_columns`` / ``id_fn`` / ``identity``.

        Raises:
            ConfigurationError: Unknown ``on_conflict`` value, or a
                duplicate-detecting policy (``error`` / ``ignore``) with no
                identity configured.
        """
        # Validate fully before any ``self.*`` mutation, mirroring
        # DatabaseBulkInsert.__init__.
        if on_conflict not in ("update", "ignore", "error"):
            raise ConfigurationError(
                f"Unknown on_conflict value '{on_conflict}' "
                "(expected 'update', 'ignore', or 'error')"
            )
        identity = resolve_identity(
            identity=identity, key_columns=key_columns, id_fn=id_fn
        )
        # ``update`` (the default) with no identity is a legitimate plain
        # create. ``error`` / ``ignore`` are explicit conflict policies that
        # need an id to detect the conflict against — without one they would
        # silently degrade to create-only (the exact advertised-but-dead knob
        # this library is closing). Fail closed, mirroring DatabaseBulkInsert.
        if on_conflict in ("error", "ignore") and identity is None:
            raise ConfigurationError(
                f"on_conflict='{on_conflict}' needs an identity to detect the "
                "conflict; pass key_columns=, id_fn=, or identity="
            )
        self.resource_name = resource_name
        self.table = table
        self.key_columns = key_columns
        self.value_columns = value_columns
        self.on_conflict = on_conflict
        self.identity = identity

    async def transform(
        self, data: Dict[str, Any], context: Any = None
    ) -> Dict[str, Any]:
        """Transform data by upserting to database.
        
        Args:
            data: Input data to upsert.
            
        Returns:
            Data with upsert result.
        """
        # Resource is injected by the engine into context.resources.
        resource = _require_resource(self.resource_name, context)
        
        # Extract record(s) to upsert
        if "records" in data:
            records = data["records"]
        elif "record" in data:
            records = [data["record"]]
        else:
            # Use the entire data as a single record
            records = [data]
        
        try:
            # Perform upsert
            result = await resource.upsert(
                table=self.table,
                records=records,
                key_columns=self.key_columns,
                value_columns=self.value_columns,
                on_conflict=self.on_conflict,
                identity=self.identity,
            )
            
            # Overrides last: ``**data`` is the passthrough, then the fresh
            # count wins over any stale ``upserted_count`` from a prior step.
            return {
                **data,
                "upserted_count": result.get("affected_rows", 0),
            }

        except (TransformError, ValidationError):
            # A row whose key columns are missing/None raises ValidationError
            # from identity derivation — surface it with its own consumer-
            # actionable type rather than masking it as a generic TransformError.
            raise
        except Exception as e:
            raise TransformError(f"Database upsert failed: {e}") from e
    
    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        return f"Upsert data into {self.table} table in {self.resource_name}"


class BatchCommit(ITransformFunction):
    """Commit a batch of records to the database."""

    def __init__(
        self,
        resource_name: str,
        batch_size: int = 1000,
        use_transaction: bool | None = None,
        *,
        key_columns: List[str] | None = None,
        id_fn: Callable[[Mapping[str, Any]], str | None] | None = None,
        identity: RecordIdentity | None = None,
        atomicity: str = "best_effort",
    ):
        """Initialize the batch commit function.

        Args:
            resource_name: Name of the database resource to use.
            batch_size: Number of records per batch.
            use_transaction: Back-compat alias for ``atomicity``. ``True`` maps
                to ``atomicity="require"`` (the commit must be all-or-nothing);
                ``False`` to ``"best_effort"``. ``None`` (default) defers to
                ``atomicity``.
            key_columns: Sugar for a :class:`KeyColumnsIdentity` — enables
                idempotent re-commit (each row upserted under its derived id).
            id_fn: Sugar for a :class:`CallableIdentity`.
            identity: An explicit :class:`RecordIdentity`. Specify at most one of
                ``key_columns`` / ``id_fn`` / ``identity``; without any, the
                batch is created (backend-assigned ids).
            atomicity: ``"best_effort"`` (default) or ``"require"`` (raise
                :class:`CapabilityNotSupportedError` when the backend cannot
                guarantee all-or-nothing).
        """
        if batch_size <= 0:
            raise ConfigurationError(
                f"batch_size must be a positive integer (got {batch_size})"
            )
        self.resource_name = resource_name
        self.batch_size = batch_size
        if use_transaction is not None:
            atomicity = "require" if use_transaction else "best_effort"
        if atomicity not in ("best_effort", "require"):
            raise ConfigurationError(
                f"Unknown atomicity policy '{atomicity}' "
                "(expected 'best_effort' or 'require')"
            )
        self.atomicity = atomicity
        self.identity = resolve_identity(
            identity=identity, key_columns=key_columns, id_fn=id_fn
        )

    async def transform(
        self, data: Dict[str, Any], context: Any = None
    ) -> Dict[str, Any]:
        """Transform data by committing batch to database.

        Args:
            data: Input data containing batch to commit.

        Returns:
            Data with commit result.
        """
        # Resource is injected by the engine into context.resources.
        resource = _require_resource(self.resource_name, context)

        # Get batch from data
        batch = data.get("batch", [])
        if not batch:
            return data

        try:
            if self.atomicity == "require":
                # A required-atomic commit must be issued as a single
                # all-or-nothing batch — chunking it would only make each chunk
                # atomic, not the whole. ``batch_size`` (a throughput knob)
                # therefore does not apply under "require": the consumer asked
                # for whole-batch atomicity, which is the unit that wins.
                result = await resource.commit_batch(
                    batch,
                    identity=self.identity,
                    atomicity=self.atomicity,
                )
                committed = result.get("affected_rows", 0)
            else:
                # best_effort: bound each commit to ``batch_size`` rows so very
                # large batches do not have to be held/sent as one unit. Each
                # chunk is committed independently (best_effort makes no
                # all-or-nothing promise across them).
                committed = 0
                for start in range(0, len(batch), self.batch_size):
                    chunk = batch[start:start + self.batch_size]
                    result = await resource.commit_batch(
                        chunk,
                        identity=self.identity,
                        atomicity=self.atomicity,
                    )
                    committed += result.get("affected_rows", 0)

            # Overrides last: ``**data`` is the passthrough, then the commit
            # outcome wins (``data`` still carries the original, now-committed
            # ``batch`` — and possibly a stale ``committed_count``).
            return {
                **data,
                "committed_count": committed,
                "batch": [],  # Clear batch after commit
            }

        except (
            TransformError,
            CapabilityNotSupportedError,
            ConfigurationError,
            ValidationError,
        ):
            # Consumer-actionable signals (atomicity policy, misconfig, a row
            # whose key columns are missing/None) surface with their own type
            # rather than being masked as a generic TransformError.
            raise
        except Exception as e:
            raise TransformError(f"Batch commit failed: {e}") from e

    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        return f"Commit batch to {self.resource_name} (batch_size={self.batch_size})"


class DatabaseQuery(ITransformFunction):
    """Execute a dynamic database query."""

    def __init__(
        self,
        resource_name: str,
        query_field: str = "query",
        params_field: str = "params",
        result_field: str = "result",
    ):
        """Initialize the database query function.
        
        Args:
            resource_name: Name of the database resource to use.
            query_field: Field containing the query to execute.
            params_field: Field containing query parameters.
            result_field: Field to store results in.
        """
        self.resource_name = resource_name
        self.query_field = query_field
        self.params_field = params_field
        self.result_field = result_field

    async def transform(
        self, data: Dict[str, Any], context: Any = None
    ) -> Dict[str, Any]:
        """Transform data by executing dynamic query.
        
        Args:
            data: Input data containing query and parameters.
            
        Returns:
            Data with query results.
        """
        # Resource is injected by the engine into context.resources.
        resource = _require_resource(self.resource_name, context)
        
        # Get query and parameters
        query = data.get(self.query_field)
        if not query:
            raise TransformError(f"Query field '{self.query_field}' not found")
        
        params = data.get(self.params_field, {})
        
        try:
            # Execute query
            result = await resource.execute_query(query, params=params)
            
            return {
                **data,
                self.result_field: result,
            }
        
        except Exception as e:
            raise TransformError(f"Query execution failed: {e}") from e
    
    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        return f"Execute dynamic query from field '{self.query_field}'"


class DatabaseTransaction(ITransformFunction):
    """Manage database transactions."""

    def __init__(
        self,
        resource_name: str,
        action: str = "begin",  # "begin", "commit", "rollback"
        savepoint: str | None = None,
    ):
        """Initialize the database transaction function.
        
        Args:
            resource_name: Name of the database resource to use.
            action: Transaction action to perform.
            savepoint: Optional savepoint name.
        """
        self.resource_name = resource_name
        self.action = action
        self.savepoint = savepoint

    async def transform(
        self, data: Dict[str, Any], context: Any = None
    ) -> Dict[str, Any]:
        """Transform data by managing transaction.
        
        Args:
            data: Input data.
            
        Returns:
            Data with transaction status.
        """
        # Resource is injected by the engine into context.resources.
        resource = _require_resource(self.resource_name, context)
        
        try:
            if self.action == "begin":
                tx = await resource.begin_transaction()
                return {
                    **data,
                    "_transaction": tx,
                    "transaction_active": True,
                }
            
            elif self.action == "commit":
                tx = data.get("_transaction")
                if tx:
                    await tx.commit()
                return {
                    **data,
                    "_transaction": None,
                    "transaction_active": False,
                }
            
            elif self.action == "rollback":
                tx = data.get("_transaction")
                if tx:
                    await tx.rollback()
                return {
                    **data,
                    "_transaction": None,
                    "transaction_active": False,
                }
            
            else:
                raise TransformError(f"Unknown action: {self.action}")
        
        except Exception as e:
            raise TransformError(f"Transaction {self.action} failed: {e}") from e
    
    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        return f"Database transaction: {self.action}"


class DatabaseBulkInsert(ITransformFunction):
    """Perform bulk insert into database."""

    def __init__(
        self,
        resource_name: str,
        table: str,
        columns: List[str] | None = None,
        chunk_size: int = 1000,
        on_duplicate: str = "error",  # "error", "ignore", "update"
        *,
        key_columns: List[str] | None = None,
        id_fn: Callable[[Mapping[str, Any]], str | None] | None = None,
        identity: RecordIdentity | None = None,
    ):
        """Initialize the bulk insert function.

        Args:
            resource_name: Name of the database resource to use.
            table: Table to insert into.
            columns: Columns to insert (if None, use all columns from first record).
            chunk_size: Number of records per chunk.
            on_duplicate: Conflict policy — ``"error"`` (raise), ``"ignore"``
                (skip), or ``"update"`` (overwrite). ``"ignore"`` / ``"update"``
                require an identity (``key_columns`` / ``id_fn`` / ``identity``)
                so duplicates can be detected.
            key_columns: Sugar for a :class:`KeyColumnsIdentity`.
            id_fn: Sugar for a :class:`CallableIdentity`.
            identity: An explicit :class:`RecordIdentity`. Specify at most one of
                ``key_columns`` / ``id_fn`` / ``identity``.

        Raises:
            ConfigurationError: Unknown ``on_duplicate`` value, or a
                duplicate-detecting policy (``ignore`` / ``update``) with no
                identity configured.
        """
        if on_duplicate not in ("error", "ignore", "update"):
            raise ConfigurationError(
                f"Unknown on_duplicate value '{on_duplicate}' "
                "(expected 'error', 'ignore', or 'update')"
            )
        if chunk_size <= 0:
            raise ConfigurationError(
                f"chunk_size must be a positive integer (got {chunk_size})"
            )
        self.resource_name = resource_name
        self.table = table
        self.columns = columns
        self.chunk_size = chunk_size
        self.on_duplicate = on_duplicate
        self.identity = resolve_identity(
            identity=identity, key_columns=key_columns, id_fn=id_fn
        )
        if on_duplicate in ("ignore", "update") and self.identity is None:
            raise ConfigurationError(
                f"on_duplicate='{on_duplicate}' needs an identity to detect "
                "duplicates; pass key_columns=, id_fn=, or identity="
            )

    async def transform(
        self, data: Dict[str, Any], context: Any = None
    ) -> Dict[str, Any]:
        """Transform data by performing bulk insert.
        
        Args:
            data: Input data containing records to insert.
            
        Returns:
            Data with insert results.
        """
        # Resource is injected by the engine into context.resources.
        resource = _require_resource(self.resource_name, context)
        
        # Get records to insert
        records = data.get("records", [])
        if not records:
            return {**data, "inserted_count": 0}
        
        # Determine columns
        columns = self.columns
        if not columns and records:
            columns = list(records[0].keys())
        
        try:
            # Perform bulk insert in chunks
            total_inserted = 0
            for i in range(0, len(records), self.chunk_size):
                chunk = records[i:i + self.chunk_size]
                result = await resource.bulk_insert(
                    table=self.table,
                    records=chunk,
                    columns=columns,
                    on_duplicate=self.on_duplicate,
                    identity=self.identity,
                )
                total_inserted += result.get("affected_rows", 0)
            
            return {
                **data,
                "inserted_count": total_inserted,
            }

        except (TransformError, ValidationError):
            # A row whose key columns are missing/None raises ValidationError
            # from identity derivation — surface it with its own consumer-
            # actionable type rather than masking it as a generic TransformError.
            raise
        except Exception as e:
            raise TransformError(f"Bulk insert failed: {e}") from e
    
    def get_transform_description(self) -> str:
        """Get a description of the transformation."""
        return f"Bulk insert into {self.table} table (chunk_size={self.chunk_size})"


# Convenience functions for creating database functions
def fetch(resource: str, query: str, **kwargs) -> DatabaseFetch:
    """Create a DatabaseFetch function."""
    return DatabaseFetch(resource, query, **kwargs)


def upsert(resource: str, table: str, keys: List[str], **kwargs) -> DatabaseUpsert:
    """Create a DatabaseUpsert function."""
    return DatabaseUpsert(resource, table, keys, **kwargs)


def commit_batch(resource: str, **kwargs) -> BatchCommit:
    """Create a BatchCommit function."""
    return BatchCommit(resource, **kwargs)


def query(resource: str, **kwargs) -> DatabaseQuery:
    """Create a DatabaseQuery function."""
    return DatabaseQuery(resource, **kwargs)


def transaction(resource: str, action: str, **kwargs) -> DatabaseTransaction:
    """Create a DatabaseTransaction function."""
    return DatabaseTransaction(resource, action, **kwargs)


def bulk_insert(resource: str, table: str, **kwargs) -> DatabaseBulkInsert:
    """Create a DatabaseBulkInsert function."""
    return DatabaseBulkInsert(resource, table, **kwargs)
