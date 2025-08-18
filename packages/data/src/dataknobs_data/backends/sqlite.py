"""SQLite backend implementation with sync and async support."""

import json
import logging
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dataknobs_config import ConfigurableBase

from ..database import SyncDatabase
from ..query import Query
from ..query_logic import ComplexQuery
from ..records import Record
from ..streaming import StreamConfig, StreamResult
from .sql_base import SQLQueryBuilder, SQLTableManager

logger = logging.getLogger(__name__)


class SyncSQLiteDatabase(SyncDatabase, ConfigurableBase):
    """Synchronous SQLite database backend."""
    
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize SQLite database.
        
        Args:
            config: Configuration with the following optional keys:
                - path: Database file path (default: ":memory:")
                - table: Table name (default: "records")
                - timeout: Connection timeout in seconds (default: 5.0)
                - check_same_thread: Allow sharing across threads (default: False)
                - journal_mode: Journal mode (WAL, DELETE, etc.) (default: None)
                - synchronous: Synchronous mode (NORMAL, FULL, OFF) (default: None)
        """
        super().__init__(config)
        self.db_path = self.config.get("path", ":memory:")
        self.table_name = self.config.get("table", "records")
        self.timeout = self.config.get("timeout", 5.0)
        self.check_same_thread = self.config.get("check_same_thread", False)
        self.journal_mode = self.config.get("journal_mode")
        self.synchronous = self.config.get("synchronous")
        
        self.query_builder = SQLQueryBuilder(self.table_name, dialect="sqlite")
        self.table_manager = SQLTableManager(self.table_name, dialect="sqlite")
        
        self.conn: sqlite3.Connection | None = None
        self._connected = False
    
    @classmethod
    def from_config(cls, config: dict) -> "SyncSQLiteDatabase":
        """Create from config dictionary."""
        return cls(config)
    
    def connect(self) -> None:
        """Connect to the SQLite database."""
        if self._connected:
            return
        
        # Create directory if needed for file-based database
        if self.db_path != ":memory:":
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self.conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=self.check_same_thread
        )
        
        # Enable row factory for dict-like access
        self.conn.row_factory = sqlite3.Row
        
        # Configure SQLite for better performance
        self._configure_sqlite()
        
        # Create table if it doesn't exist
        self._ensure_table()
        
        self._connected = True
        logger.info(f"Connected to SQLite database: {self.db_path}")
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self._connected = False
            logger.info(f"Disconnected from SQLite database: {self.db_path}")
    
    def _configure_sqlite(self) -> None:
        """Configure SQLite settings for performance."""
        if not self.conn:
            return
        
        cursor = self.conn.cursor()
        
        # Set journal mode if specified
        if self.journal_mode:
            cursor.execute(f"PRAGMA journal_mode = {self.journal_mode}")
            logger.debug(f"Set journal_mode to {self.journal_mode}")
        
        # Set synchronous mode if specified
        if self.synchronous:
            cursor.execute(f"PRAGMA synchronous = {self.synchronous}")
            logger.debug(f"Set synchronous to {self.synchronous}")
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Optimize for performance
        cursor.execute("PRAGMA temp_store = MEMORY")
        cursor.execute("PRAGMA mmap_size = 30000000000")
        
        cursor.close()
    
    def _ensure_table(self) -> None:
        """Ensure the table exists."""
        if not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        cursor = self.conn.cursor()
        cursor.executescript(self.table_manager.get_create_table_sql())
        self.conn.commit()
        cursor.close()
    
    def _check_connection(self) -> None:
        """Check if database is connected."""
        if not self._connected or not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")
    
    def create(self, record: Record) -> str:
        """Create a new record."""
        self._check_connection()
        
        query, params = self.query_builder.build_create_query(record)
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(query, params)
            self.conn.commit()
            
            # SQLite doesn't support RETURNING, so we use the ID we generated
            record_id = params[0]  # ID is the first parameter
            return record_id
        except sqlite3.IntegrityError:
            self.conn.rollback()
            raise ValueError(f"Record with ID {params[0]} already exists")
        finally:
            cursor.close()
    
    def read(self, id: str) -> Record | None:
        """Read a record by ID."""
        self._check_connection()
        
        query, params = self.query_builder.build_read_query(id)
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            if row:
                return SQLQueryBuilder.row_to_record(dict(row))
            return None
        finally:
            cursor.close()
    
    def update(self, id: str, record: Record) -> bool:
        """Update an existing record."""
        self._check_connection()
        
        query, params = self.query_builder.build_update_query(id, record)
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(query, params)
            self.conn.commit()
            return cursor.rowcount > 0
        finally:
            cursor.close()
    
    def delete(self, id: str) -> bool:
        """Delete a record by ID."""
        self._check_connection()
        
        query, params = self.query_builder.build_delete_query(id)
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(query, params)
            self.conn.commit()
            return cursor.rowcount > 0
        finally:
            cursor.close()
    
    def exists(self, id: str) -> bool:
        """Check if a record exists."""
        self._check_connection()
        
        query, params = self.query_builder.build_exists_query(id)
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result is not None
        finally:
            cursor.close()
    
    def search(self, query: Query | ComplexQuery) -> list[Record]:
        """Search for records matching a query."""
        self._check_connection()
        
        # Handle ComplexQuery with native SQL support
        if isinstance(query, ComplexQuery):
            sql_query, params = self.query_builder.build_complex_search_query(query)
        else:
            sql_query, params = self.query_builder.build_search_query(query)
        
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(sql_query, params)
            rows = cursor.fetchall()
            
            records = [SQLQueryBuilder.row_to_record(dict(row)) for row in rows]
            
            # Apply field projection if specified
            if query.fields:
                records = [r.project(query.fields) for r in records]
            
            return records
        finally:
            cursor.close()
    
    def count(self, query: Query | None = None) -> int:
        """Count records matching a query."""
        self._check_connection()
        
        sql_query, params = self.query_builder.build_count_query(query)
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(sql_query, params)
            result = cursor.fetchone()
            return result[0] if result else 0
        finally:
            cursor.close()
    
    def create_batch(self, records: list[Record]) -> list[str]:
        """Create multiple records in a batch."""
        self._check_connection()
        
        ids = []
        cursor = self.conn.cursor()
        
        try:
            # Use a transaction for better performance
            cursor.execute("BEGIN TRANSACTION")
            
            for record in records:
                query, params = self.query_builder.build_create_query(record)
                cursor.execute(query, params)
                ids.append(params[0])  # ID is the first parameter
            
            self.conn.commit()
            return ids
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()
    
    def update_batch(self, updates: list[tuple[str, Record]]) -> list[bool]:
        """Update multiple records in a batch."""
        self._check_connection()
        
        results = []
        cursor = self.conn.cursor()
        
        try:
            # Use a transaction for better performance
            cursor.execute("BEGIN TRANSACTION")
            
            for id, record in updates:
                query, params = self.query_builder.build_update_query(id, record)
                cursor.execute(query, params)
                results.append(cursor.rowcount > 0)
            
            self.conn.commit()
            return results
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()
    
    def delete_batch(self, ids: list[str]) -> list[bool]:
        """Delete multiple records in a batch."""
        self._check_connection()
        
        results = []
        cursor = self.conn.cursor()
        
        try:
            # Use a transaction for better performance
            cursor.execute("BEGIN TRANSACTION")
            
            for id in ids:
                query, params = self.query_builder.build_delete_query(id)
                cursor.execute(query, params)
                results.append(cursor.rowcount > 0)
            
            self.conn.commit()
            return results
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()
    
    def _initialize(self) -> None:
        """Initialize method - connection setup handled in connect()."""
        pass
    
    def _count_all(self) -> int:
        """Count all records in the database."""
        self._check_connection()
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            result = cursor.fetchone()
            return result[0] if result else 0
        finally:
            cursor.close()
    
    def stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> Iterator[Record]:
        """Stream records from database."""
        from ..streaming import StreamConfig
        
        config = config or StreamConfig()
        query = query or Query()
        
        # Use the existing stream method's logic but yield individual records
        offset = 0
        while True:
            # Fetch a batch
            query_copy = query.copy()
            query_copy.offset(offset).limit(config.batch_size)
            batch = self.search(query_copy)
            
            if not batch:
                break
                
            for record in batch:
                yield record
            
            offset += len(batch)
            
            # If we got less than batch_size, we're done
            if len(batch) < config.batch_size:
                break
    
    def stream_write(
        self,
        records: Iterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Stream records into database."""
        from ..streaming import StreamConfig, StreamResult, StreamProgress
        import time
        
        config = config or StreamConfig()
        batch = []
        total_written = 0
        start_time = time.time()
        
        for record in records:
            batch.append(record)
            
            if len(batch) >= config.batch_size:
                # Write the batch
                self.create_batch(batch)
                total_written += len(batch)
                batch = []
        
        # Write any remaining records
        if batch:
            self.create_batch(batch)
            total_written += len(batch)
        
        elapsed = time.time() - start_time
        
        return StreamResult(
            records=[],  # Don't return records for write operations
            has_more=False,
            progress=StreamProgress(
                current=total_written,
                total=total_written,
                percentage=100.0
            ),
            metadata={
                "total_written": total_written,
                "elapsed_seconds": elapsed
            }
        )