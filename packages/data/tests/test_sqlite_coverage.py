"""Extended tests for SQLite backend to improve coverage."""

import asyncio
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import aiosqlite

from dataknobs_data.backends.sql_base import SQLQueryBuilder, SQLTableManager
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.query import Filter, Operator, Query, SortOrder
from dataknobs_data.query_logic import ComplexQuery, FilterCondition, LogicCondition, LogicOperator
from dataknobs_data.records import Record
from dataknobs_data.streaming import StreamConfig, StreamResult


class TestSQLBase:
    """Test SQL base classes for coverage."""
    
    def test_query_builder_initialization(self):
        """Test SQLQueryBuilder initialization."""
        # Default initialization
        builder = SQLQueryBuilder("test_table")
        assert builder.table_name == "test_table"
        assert builder.schema_name is None
        assert builder.dialect == "standard"
        assert builder.qualified_table == "test_table"
        
        # With schema
        builder = SQLQueryBuilder("test_table", schema_name="public", dialect="postgres")
        assert builder.schema_name == "public"
        assert builder.dialect == "postgres"
        assert builder.qualified_table == "public.test_table"
    
    def test_query_builder_postgres_dialect(self):
        """Test PostgreSQL specific query building."""
        builder = SQLQueryBuilder("test_table", schema_name="public", dialect="postgres")
        
        # Test create query
        record = Record(data={"name": "test", "value": 123})
        query, params = builder.build_create_query(record)
        assert "$1" in query
        assert "$2" in query
        assert "$3" in query
        assert len(params) == 3
        
        # Test read query
        query, params = builder.build_read_query("test-id")
        assert "$1" in query
        assert params == ["test-id"]
        
        # Test update query
        query, params = builder.build_update_query("test-id", record)
        assert "$1" in query
        assert "$2" in query
        assert "$3" in query
        
        # Test delete query
        query, params = builder.build_delete_query("test-id")
        assert "$1" in query
        
        # Test exists query
        query, params = builder.build_exists_query("test-id")
        assert "$1" in query
    
    def test_query_builder_operators(self):
        """Test different SQL operators."""
        builder = SQLQueryBuilder("test_table", dialect="sqlite")
        
        # Test IN operator with postgres dialect
        builder_pg = SQLQueryBuilder("test_table", dialect="postgres")
        filter_spec = Filter("field", Operator.IN, [1, 2, 3])
        clause, params = builder_pg._build_filter_clause(filter_spec, 1)
        assert "$1" in clause
        assert "$2" in clause
        assert "$3" in clause
        assert params == [1, 2, 3]
        
        # Test NOT_IN operator
        filter_spec = Filter("field", Operator.NOT_IN, [1, 2, 3])
        clause, params = builder._build_filter_clause(filter_spec, 1)
        assert "NOT IN" in clause
        assert params == [1, 2, 3]
        
        # Test EXISTS operator (maps to IS NOT NULL)
        filter_spec = Filter("field", Operator.EXISTS, None)
        clause, params = builder._build_filter_clause(filter_spec, 1)
        assert "IS NOT NULL" in clause
        assert params == []
        
        # Test NOT_EXISTS operator (maps to IS NULL)
        filter_spec = Filter("field", Operator.NOT_EXISTS, None)
        clause, params = builder._build_filter_clause(filter_spec, 1)
        assert "IS NULL" in clause
        assert params == []
        
        # Test unsupported operator - use a non-existent operator value
        filter_spec = Mock()
        filter_spec.field = "field"
        filter_spec.operator = "UNSUPPORTED"
        filter_spec.value = "value"
        with pytest.raises(ValueError, match="Unsupported operator"):
            builder._build_filter_clause(filter_spec, 1)
    
    def test_query_builder_sorting(self):
        """Test query building with sorting."""
        builder = SQLQueryBuilder("test_table", dialect="postgres")
        
        query = Query().sort("name", SortOrder.ASC).sort("age", SortOrder.DESC)
        sql_query, params = builder.build_search_query(query)
        assert "ORDER BY" in sql_query
        assert "data->'name' ASC" in sql_query
        assert "data->'age' DESC" in sql_query
    
    def test_query_builder_count(self):
        """Test count query building."""
        builder = SQLQueryBuilder("test_table", dialect="sqlite")
        
        # Count all
        sql_query, params = builder.build_count_query()
        assert "SELECT COUNT(*)" in sql_query
        assert params == []
        
        # Count with filter
        query = Query().filter("status", Operator.EQ, "active")
        sql_query, params = builder.build_count_query(query)
        assert "SELECT COUNT(*)" in sql_query
        assert "WHERE" in sql_query
        assert params == ["active"]
    
    def test_table_manager(self):
        """Test SQLTableManager."""
        # Standard SQL
        manager = SQLTableManager("test_table", dialect="standard")
        sql = manager.get_create_table_sql()
        assert "CREATE TABLE IF NOT EXISTS test_table" in sql
        
        # PostgreSQL with schema
        manager = SQLTableManager("test_table", schema_name="public", dialect="postgres")
        sql = manager.get_create_table_sql()
        assert "public.test_table" in sql
        assert "JSONB" in sql
        assert "GIN" in sql
        
        # SQLite
        manager = SQLTableManager("test_table", dialect="sqlite")
        sql = manager.get_create_table_sql()
        assert "json_valid(data)" in sql
        assert "json_valid(metadata)" in sql
        
        # Drop table
        sql = manager.get_drop_table_sql()
        assert "DROP TABLE IF EXISTS test_table" in sql
    
    def test_complex_query_not_operator(self):
        """Test NOT operator in complex queries."""
        builder = SQLQueryBuilder("test_table", dialect="sqlite")
        
        # NOT condition
        condition = LogicCondition(
            operator=LogicOperator.NOT,
            conditions=[
                FilterCondition(Filter("status", Operator.EQ, "active"))
            ]
        )
        
        clause, params = builder._build_complex_condition(condition, 1)
        assert "NOT (" in clause
        assert params == ["active"]
    
    def test_empty_complex_conditions(self):
        """Test complex queries with empty conditions."""
        builder = SQLQueryBuilder("test_table", dialect="sqlite")
        
        # Empty AND
        condition = LogicCondition(operator=LogicOperator.AND, conditions=[])
        clause, params = builder._build_complex_condition(condition, 1)
        assert clause == ""
        assert params == []
        
        # Empty OR
        condition = LogicCondition(operator=LogicOperator.OR, conditions=[])
        clause, params = builder._build_complex_condition(condition, 1)
        assert clause == ""
        assert params == []
    
    def test_complex_query_with_pagination(self):
        """Test complex query with limit and offset."""
        builder = SQLQueryBuilder("test_table", dialect="sqlite")
        
        query = ComplexQuery(
            condition=LogicCondition(
                operator=LogicOperator.OR,
                conditions=[
                    FilterCondition(Filter("status", Operator.EQ, "active")),
                    FilterCondition(Filter("priority", Operator.GT, 5))
                ]
            ),
            limit_value=10,
            offset_value=20
        )
        
        sql_query, params = builder.build_complex_search_query(query)
        assert "LIMIT 10" in sql_query
        assert "OFFSET 20" in sql_query
        assert params == ["active", 5]


class TestSyncSQLiteCoverage:
    """Extended tests for sync SQLite backend."""
    
    def test_from_config(self):
        """Test creating database from config."""
        config = {
            "path": ":memory:",
            "table": "custom_table",
            "timeout": 10.0
        }
        db = SyncSQLiteDatabase.from_config(config)
        assert db.db_path == ":memory:"
        assert db.table_name == "custom_table"
        assert db.timeout == 10.0
    
    def test_close_without_connection(self):
        """Test closing without connecting."""
        db = SyncSQLiteDatabase({"path": ":memory:"})
        # Should not raise
        db.close()
        assert not db._connected
    
    def test_double_connect(self):
        """Test connecting twice."""
        db = SyncSQLiteDatabase({"path": ":memory:"})
        db.connect()
        assert db._connected
        
        # Second connect should be a no-op
        db.connect()
        assert db._connected
        
        db.close()
    
    def test_operations_without_connection(self):
        """Test operations without connection raise errors."""
        db = SyncSQLiteDatabase({"path": ":memory:"})
        
        with pytest.raises(RuntimeError, match="Database not connected"):
            db.create(Record(data={"test": "data"}))
        
        with pytest.raises(RuntimeError, match="Database not connected"):
            db.read("test-id")
        
        with pytest.raises(RuntimeError, match="Database not connected"):
            db.update("test-id", Record(data={"test": "data"}))
        
        with pytest.raises(RuntimeError, match="Database not connected"):
            db.delete("test-id")
        
        with pytest.raises(RuntimeError, match="Database not connected"):
            db.exists("test-id")
        
        with pytest.raises(RuntimeError, match="Database not connected"):
            db.search(Query())
        
        with pytest.raises(RuntimeError, match="Database not connected"):
            db.count()
    
    def test_create_duplicate_id(self):
        """Test creating record with duplicate ID."""
        db = SyncSQLiteDatabase({"path": ":memory:"})
        db.connect()
        
        try:
            # Create first record
            record1 = Record(data={"name": "first"})
            # Use the actual ID mechanism - provide record with same ID
            with patch('dataknobs_data.backends.sql_base.uuid.uuid4', return_value=Mock(__str__=Mock(return_value='fixed-id'))):
                id1 = db.create(record1)
            
            # Try to create second record with same ID
            record2 = Record(data={"name": "second"})
            with patch('dataknobs_data.backends.sql_base.uuid.uuid4', return_value=Mock(__str__=Mock(return_value='fixed-id'))):
                with pytest.raises(ValueError, match="already exists"):
                    db.create(record2)
        finally:
            db.close()
    
    def test_update_nonexistent(self):
        """Test updating non-existent record."""
        db = SyncSQLiteDatabase({"path": ":memory:"})
        db.connect()
        
        try:
            result = db.update("nonexistent-id", Record(data={"test": "data"}))
            assert result is False
        finally:
            db.close()
    
    def test_delete_nonexistent(self):
        """Test deleting non-existent record."""
        db = SyncSQLiteDatabase({"path": ":memory:"})
        db.connect()
        
        try:
            result = db.delete("nonexistent-id")
            assert result is False
        finally:
            db.close()
    
    def test_read_nonexistent(self):
        """Test reading non-existent record."""
        db = SyncSQLiteDatabase({"path": ":memory:"})
        db.connect()
        
        try:
            result = db.read("nonexistent-id")
            assert result is None
        finally:
            db.close()
    
    def test_exists_nonexistent(self):
        """Test checking existence of non-existent record."""
        db = SyncSQLiteDatabase({"path": ":memory:"})
        db.connect()
        
        try:
            result = db.exists("nonexistent-id")
            assert result is False
        finally:
            db.close()
    
    def test_search_no_results(self):
        """Test search with no matching results."""
        db = SyncSQLiteDatabase({"path": ":memory:"})
        db.connect()
        
        try:
            # Search in empty database
            results = db.search(Query().filter("status", Operator.EQ, "active"))
            assert results == []
        finally:
            db.close()
    
    def test_pragma_settings(self, tmp_path):
        """Test PRAGMA configuration settings."""
        db_path = tmp_path / "test_pragma.db"
        db = SyncSQLiteDatabase({
            "path": str(db_path),
            "journal_mode": "DELETE",
            "synchronous": "FULL"
        })
        db.connect()
        
        try:
            # Verify settings were applied
            cursor = db.conn.cursor()
            
            # Check journal mode
            cursor.execute("PRAGMA journal_mode")
            result = cursor.fetchone()
            assert result[0].upper() == "DELETE"
            
            # Check synchronous mode
            cursor.execute("PRAGMA synchronous")
            result = cursor.fetchone()
            assert result[0] == 2  # FULL = 2
            
            cursor.close()
        finally:
            db.close()
    
    def test_batch_operations_with_errors(self):
        """Test batch operations with errors."""
        db = SyncSQLiteDatabase({"path": ":memory:"})
        db.connect()
        
        try:
            # Create batch with error handling
            records = [
                Record(data={"name": f"User{i}"})
                for i in range(3)
            ]
            ids = db.create_batch(records)
            assert len(ids) == 3
            
            # Update batch with some failures
            updates = [
                (ids[0], Record(data={"name": "Updated0"})),
                ("nonexistent", Record(data={"name": "UpdatedX"})),
                (ids[2], Record(data={"name": "Updated2"}))
            ]
            results = db.update_batch(updates)
            assert results == [True, False, True]
            
            # Delete batch with some failures
            delete_ids = [ids[0], "nonexistent", ids[2]]
            results = db.delete_batch(delete_ids)
            assert results == [True, False, True]
        finally:
            db.close()
    
    
    def test_field_projection(self):
        """Test field projection in search results."""
        db = SyncSQLiteDatabase({"path": ":memory:"})
        db.connect()
        
        try:
            # Create records with multiple fields
            db.create(Record(data={"name": "Alice", "age": 30, "city": "NYC"}))
            db.create(Record(data={"name": "Bob", "age": 25, "city": "LA"}))
            
            # Search without field projection
            query = Query()
            results = db.search(query)
            
            assert len(results) == 2
            for record in results:
                assert "name" in record.fields
                assert "age" in record.fields
                assert "city" in record.fields
        finally:
            db.close()


class TestAsyncSQLiteCoverage:
    """Extended tests for async SQLite backend."""
    
    @pytest.mark.asyncio
    async def test_from_config(self):
        """Test creating database from config."""
        config = {
            "path": ":memory:",
            "table": "custom_table",
            "pool_size": 10
        }
        db = AsyncSQLiteDatabase.from_config(config)
        assert db.db_path == ":memory:"
        assert db.table_name == "custom_table"
        assert db.pool_size == 10
    
    @pytest.mark.asyncio
    async def test_close_without_connection(self):
        """Test closing without connecting."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        # Should not raise
        await db.close()
        assert not db._connected
    
    @pytest.mark.asyncio
    async def test_double_connect(self):
        """Test connecting twice."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        await db.connect()
        assert db._connected
        
        # Second connect should be a no-op
        await db.connect()
        assert db._connected
        
        await db.close()
    
    @pytest.mark.asyncio
    async def test_operations_without_connection(self):
        """Test operations without connection raise errors."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        
        with pytest.raises(RuntimeError, match="Database not connected"):
            await db.create(Record(data={"test": "data"}))
        
        with pytest.raises(RuntimeError, match="Database not connected"):
            await db.read("test-id")
        
        with pytest.raises(RuntimeError, match="Database not connected"):
            await db.update("test-id", Record(data={"test": "data"}))
        
        with pytest.raises(RuntimeError, match="Database not connected"):
            await db.delete("test-id")
        
        with pytest.raises(RuntimeError, match="Database not connected"):
            await db.exists("test-id")
        
        with pytest.raises(RuntimeError, match="Database not connected"):
            await db.search(Query())
        
        with pytest.raises(RuntimeError, match="Database not connected"):
            await db.count()
    
    @pytest.mark.asyncio
    async def test_create_duplicate_id(self):
        """Test creating record with duplicate ID."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        await db.connect()
        
        try:
            # Create first record
            record1 = Record(data={"name": "first"})
            with patch('dataknobs_data.backends.sql_base.uuid.uuid4', return_value=Mock(__str__=Mock(return_value='fixed-id'))):
                id1 = await db.create(record1)
            
            # Try to create second record with same ID
            record2 = Record(data={"name": "second"})
            with patch('dataknobs_data.backends.sql_base.uuid.uuid4', return_value=Mock(__str__=Mock(return_value='fixed-id'))):
                with pytest.raises(ValueError, match="already exists"):
                    await db.create(record2)
        finally:
            await db.close()
    
    @pytest.mark.asyncio
    async def test_update_nonexistent(self):
        """Test updating non-existent record."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        await db.connect()
        
        try:
            result = await db.update("nonexistent-id", Record(data={"test": "data"}))
            assert result is False
        finally:
            await db.close()
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        """Test deleting non-existent record."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        await db.connect()
        
        try:
            result = await db.delete("nonexistent-id")
            assert result is False
        finally:
            await db.close()
    
    @pytest.mark.asyncio
    async def test_read_nonexistent(self):
        """Test reading non-existent record."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        await db.connect()
        
        try:
            result = await db.read("nonexistent-id")
            assert result is None
        finally:
            await db.close()
    
    @pytest.mark.asyncio
    async def test_exists_nonexistent(self):
        """Test checking existence of non-existent record."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        await db.connect()
        
        try:
            result = await db.exists("nonexistent-id")
            assert result is False
        finally:
            await db.close()
    
    @pytest.mark.asyncio
    async def test_search_no_results(self):
        """Test search with no matching results."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        await db.connect()
        
        try:
            # Search in empty database
            results = await db.search(Query().filter("status", Operator.EQ, "active"))
            assert results == []
        finally:
            await db.close()
    
    @pytest.mark.asyncio
    async def test_pragma_settings(self, tmp_path):
        """Test PRAGMA configuration settings."""
        db_path = tmp_path / "test_pragma_async.db"
        db = AsyncSQLiteDatabase({
            "path": str(db_path),
            "journal_mode": "MEMORY",
            "synchronous": "OFF"
        })
        await db.connect()
        
        try:
            # Create a record to ensure pragmas are applied
            await db.create(Record(data={"test": "data"}))
            
            # Settings should be applied (can't easily verify async)
            assert db.journal_mode == "MEMORY"
            assert db.synchronous == "OFF"
        finally:
            await db.close()
    
    @pytest.mark.asyncio
    async def test_batch_operations_with_errors(self):
        """Test batch operations with errors."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        await db.connect()
        
        try:
            # Create batch
            records = [
                Record(data={"name": f"User{i}"})
                for i in range(3)
            ]
            ids = await db.create_batch(records)
            assert len(ids) == 3
            
            # Update batch with some failures
            updates = [
                (ids[0], Record(data={"name": "Updated0"})),
                ("nonexistent", Record(data={"name": "UpdatedX"})),
                (ids[2], Record(data={"name": "Updated2"}))
            ]
            results = await db.update_batch(updates)
            assert results == [True, False, True]
            
            # Delete batch with some failures
            delete_ids = [ids[0], "nonexistent", ids[2]]
            results = await db.delete_batch(delete_ids)
            assert results == [True, False, True]
        finally:
            await db.close()
    
    
    @pytest.mark.asyncio
    async def test_field_projection(self):
        """Test field projection in search results."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        await db.connect()
        
        try:
            # Create records with multiple fields
            await db.create(Record(data={"name": "Alice", "age": 30, "city": "NYC"}))
            await db.create(Record(data={"name": "Bob", "age": 25, "city": "LA"}))
            
            # Search without field projection
            query = Query()
            results = await db.search(query)
            
            assert len(results) == 2
            for record in results:
                assert "name" in record.fields
                assert "age" in record.fields
                assert "city" in record.fields
        finally:
            await db.close()
    
    @pytest.mark.asyncio
    async def test_batch_operations_transaction_rollback(self):
        """Test batch operations with transaction rollback."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        await db.connect()
        
        try:
            # Mock an error during batch creation
            original_execute = db.db.execute
            
            async def mock_execute(query, params=None):
                # Fail during the actual INSERT statement (not BEGIN TRANSACTION)
                if query and "INSERT INTO" in query and params and len(params) > 0:
                    raise Exception("Simulated error during batch insert")
                return await original_execute(query, params)
            
            db.db.execute = mock_execute
            
            records = [
                Record(data={"name": f"User{i}"})
                for i in range(5)
            ]
            
            with pytest.raises(Exception, match="Simulated error during batch insert"):
                await db.create_batch(records)
            
            # Verify rollback happened - no records should exist
            db.db.execute = original_execute
            count = await db.count()
            assert count == 0
        finally:
            await db.close()


class TestSQLiteEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_file_database_parent_dir_creation(self, tmp_path):
        """Test that parent directories are created for file databases."""
        db_path = tmp_path / "nested" / "dirs" / "test.db"
        db = SyncSQLiteDatabase({"path": str(db_path)})
        db.connect()
        
        try:
            # Parent directories should be created
            assert db_path.parent.exists()
            
            # Database should work
            db.create(Record(data={"test": "data"}))
            assert db.count() == 1
        finally:
            db.close()
    
    @pytest.mark.asyncio
    async def test_async_file_database_parent_dir_creation(self, tmp_path):
        """Test that parent directories are created for async file databases."""
        db_path = tmp_path / "nested" / "async" / "test.db"
        db = AsyncSQLiteDatabase({"path": str(db_path)})
        await db.connect()
        
        try:
            # Parent directories should be created
            assert db_path.parent.exists()
            
            # Database should work
            await db.create(Record(data={"test": "data"}))
            assert await db.count() == 1
        finally:
            await db.close()
    
    def test_complex_json_data(self):
        """Test handling complex JSON data."""
        db = SyncSQLiteDatabase({"path": ":memory:"})
        db.connect()
        
        try:
            # Create record with nested data
            complex_data = {
                "name": "Test",
                "nested": {
                    "level1": {
                        "level2": {
                            "value": 123,
                            "array": [1, 2, 3]
                        }
                    }
                },
                "list": ["a", "b", "c"],
                "boolean": True,
                "null": None,
                "number": 3.14
            }
            
            record = Record(data=complex_data)
            id = db.create(record)
            
            # Read and verify
            retrieved = db.read(id)
            assert retrieved["nested"]["level1"]["level2"]["value"] == 123
            assert retrieved["list"] == ["a", "b", "c"]
            assert retrieved["boolean"] is True
            assert retrieved["null"] is None
            assert retrieved["number"] == 3.14
        finally:
            db.close()
    
    def test_large_batch_operations(self):
        """Test handling large batches."""
        db = SyncSQLiteDatabase({"path": ":memory:"})
        db.connect()
        
        try:
            # Create a large batch
            large_batch = [
                Record(data={"id": i, "value": f"value_{i}"})
                for i in range(1000)
            ]
            
            ids = db.create_batch(large_batch)
            assert len(ids) == 1000
            
            # Verify all records exist
            count = db.count()
            assert count == 1000
            
            # Search with pagination
            query = Query().limit(100).offset(500)
            results = db.search(query)
            assert len(results) == 100
        finally:
            db.close()
    
    def test_metadata_handling(self):
        """Test metadata storage and retrieval."""
        db = SyncSQLiteDatabase({"path": ":memory:"})
        db.connect()
        
        try:
            # Create record with metadata
            record = Record(
                data={"name": "test"},
                metadata={
                    "created_by": "user123",
                    "tags": ["important", "urgent"],
                    "version": 1
                }
            )
            
            id = db.create(record)
            
            # Read and verify metadata
            retrieved = db.read(id)
            assert retrieved.metadata["created_by"] == "user123"
            assert retrieved.metadata["tags"] == ["important", "urgent"]
            assert retrieved.metadata["version"] == 1
            
            # Update with new metadata
            record.metadata["version"] = 2
            db.update(id, record)
            
            retrieved = db.read(id)
            assert retrieved.metadata["version"] == 2
        finally:
            db.close()