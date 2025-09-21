"""Built-in database functions for FSM.

This module provides database-related functions that can be referenced
in FSM configurations, leveraging the dataknobs_data package.
"""

from typing import Any, Dict, List

from dataknobs_fsm.functions.base import ITransformFunction, TransformError
from dataknobs_fsm.resources.database import DatabaseResourceAdapter


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

    async def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by fetching from database.
        
        Args:
            data: Input data (can contain query parameters).
            
        Returns:
            Data with database query results.
        """
        # Get resource from context (injected during execution)
        resource = data.get("_resources", {}).get(self.resource_name)
        if not resource or not isinstance(resource, DatabaseResourceAdapter):
            raise TransformError(
                f"Database resource '{self.resource_name}' not found"
            )
        
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
        key_columns: List[str],
        value_columns: List[str] | None = None,
        on_conflict: str = "update",  # "update", "ignore", "error"
    ):
        """Initialize the database upsert function.
        
        Args:
            resource_name: Name of the database resource to use.
            table: Table name to upsert into.
            key_columns: Columns that form the unique key.
            value_columns: Columns to update (if None, update all non-key columns).
            on_conflict: Action on conflict ("update", "ignore", "error").
        """
        self.resource_name = resource_name
        self.table = table
        self.key_columns = key_columns
        self.value_columns = value_columns
        self.on_conflict = on_conflict

    async def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by upserting to database.
        
        Args:
            data: Input data to upsert.
            
        Returns:
            Data with upsert result.
        """
        # Get resource from context
        resource = data.get("_resources", {}).get(self.resource_name)
        if not resource or not isinstance(resource, DatabaseResourceAdapter):
            raise TransformError(
                f"Database resource '{self.resource_name}' not found"
            )
        
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
            )
            
            return {
                "upserted_count": result.get("affected_rows", 0),
                **data,
            }
        
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
        use_transaction: bool = True,
    ):
        """Initialize the batch commit function.
        
        Args:
            resource_name: Name of the database resource to use.
            batch_size: Number of records per batch.
            use_transaction: Whether to use a transaction for each batch.
        """
        self.resource_name = resource_name
        self.batch_size = batch_size
        self.use_transaction = use_transaction

    async def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by committing batch to database.
        
        Args:
            data: Input data containing batch to commit.
            
        Returns:
            Data with commit result.
        """
        # Get resource from context
        resource = data.get("_resources", {}).get(self.resource_name)
        if not resource or not isinstance(resource, DatabaseResourceAdapter):
            raise TransformError(
                f"Database resource '{self.resource_name}' not found"
            )
        
        # Get batch from data
        batch = data.get("batch", [])
        if not batch:
            return data
        
        try:
            if self.use_transaction:
                # Commit with transaction
                async with resource.transaction() as tx:
                    await tx.commit_batch(batch)
                    committed = len(batch)
            else:
                # Direct commit
                result = await resource.commit_batch(batch)
                committed = result.get("affected_rows", 0)
            
            return {
                "committed_count": committed,
                "batch": [],  # Clear batch after commit
                **data,
            }
        
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

    async def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by executing dynamic query.
        
        Args:
            data: Input data containing query and parameters.
            
        Returns:
            Data with query results.
        """
        # Get resource from context
        resource = data.get("_resources", {}).get(self.resource_name)
        if not resource or not isinstance(resource, DatabaseResourceAdapter):
            raise TransformError(
                f"Database resource '{self.resource_name}' not found"
            )
        
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

    async def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by managing transaction.
        
        Args:
            data: Input data.
            
        Returns:
            Data with transaction status.
        """
        # Get resource from context
        resource = data.get("_resources", {}).get(self.resource_name)
        if not resource or not isinstance(resource, DatabaseResourceAdapter):
            raise TransformError(
                f"Database resource '{self.resource_name}' not found"
            )
        
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
    ):
        """Initialize the bulk insert function.
        
        Args:
            resource_name: Name of the database resource to use.
            table: Table to insert into.
            columns: Columns to insert (if None, use all columns from first record).
            chunk_size: Number of records per chunk.
            on_duplicate: Action on duplicate key.
        """
        self.resource_name = resource_name
        self.table = table
        self.columns = columns
        self.chunk_size = chunk_size
        self.on_duplicate = on_duplicate

    async def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data by performing bulk insert.
        
        Args:
            data: Input data containing records to insert.
            
        Returns:
            Data with insert results.
        """
        # Get resource from context
        resource = data.get("_resources", {}).get(self.resource_name)
        if not resource or not isinstance(resource, DatabaseResourceAdapter):
            raise TransformError(
                f"Database resource '{self.resource_name}' not found"
            )
        
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
                )
                total_inserted += result.get("affected_rows", 0)
            
            return {
                **data,
                "inserted_count": total_inserted,
            }
        
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
