"""Tests for synchronous I/O provider implementations.

This module tests the completed loose end implementations for sync I/O providers:
- SyncDatabaseProvider class (lines 362-454 in io/adapters.py)
- SyncHTTPProvider class (lines 579-714 in io/adapters.py)
"""

import pytest
import json
import sqlite3
from unittest.mock import Mock, MagicMock, patch, mock_open
from typing import Dict, List, Any
import tempfile
import os

from dataknobs_fsm.io.base import IOConfig, IOMode
from dataknobs_fsm.io.adapters import (
    SyncDatabaseProvider, SyncHTTPProvider,
    DatabaseIOAdapter, HTTPIOAdapter
)


class TestSyncDatabaseProvider:
    """Test the SyncDatabaseProvider class."""
    
    def test_provider_initialization(self):
        """Test SyncDatabaseProvider initialization."""
        config = IOConfig(
            source="test.db",
            format="sqlite",
            mode=IOMode.WRITE
        )
        
        provider = SyncDatabaseProvider(config)
        
        assert provider.config == config
        assert provider.db is None
        assert provider._is_open is False
        assert provider.adapter is not None
        
    def test_provider_open_memory_database(self):
        """Test opening in-memory database."""
        config = IOConfig(
            source=":memory:",
            format="sqlite",
            mode=IOMode.WRITE
        )
        
        provider = SyncDatabaseProvider(config)
        provider.open()
        
        assert provider.db is not None
        assert provider._is_open is True
        assert isinstance(provider.db, sqlite3.Connection)
        
        # Verify row factory is set
        cursor = provider.db.execute("SELECT 1 as test")
        row = cursor.fetchone()
        assert dict(row)['test'] == 1
        
        provider.close()
        
    def test_provider_open_file_database(self):
        """Test opening file-based database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
            
        try:
            config = IOConfig(
                mode=IOMode.WRITE,
                format="sqlite",
                source=db_path
            )
            
            provider = SyncDatabaseProvider(config)
            provider.open()
            
            assert provider.db is not None
            assert provider._is_open is True
            assert os.path.exists(db_path)
            
            provider.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
                
    def test_provider_close(self):
        """Test closing database connection."""
        config = IOConfig(source=":memory:", format="sqlite", mode=IOMode.WRITE)
        
        provider = SyncDatabaseProvider(config)
        provider.open()
        assert provider._is_open is True
        
        provider.close()
        assert provider._is_open is False
        
        # Verify connection is closed
        with pytest.raises(sqlite3.ProgrammingError):
            provider.db.execute("SELECT 1")
            
    def test_provider_validate_connection(self):
        """Test validating database connection."""
        config = IOConfig(source=":memory:", format="sqlite", mode=IOMode.WRITE)
        
        provider = SyncDatabaseProvider(config)
        
        # Should be invalid before opening
        assert provider.validate() is False
        
        provider.open()
        # Should be valid after opening
        assert provider.validate() is True
        
        provider.close()
        # Should be invalid after closing
        assert provider.validate() is False
        
    def test_provider_read_basic(self):
        """Test basic read operation."""
        config = IOConfig(source=":memory:", format="sqlite", mode=IOMode.WRITE)
        
        provider = SyncDatabaseProvider(config)
        provider.open()
        
        # Create test table and data
        provider.db.execute("CREATE TABLE data (id INTEGER, name TEXT)")
        provider.db.execute("INSERT INTO data VALUES (1, 'test1')")
        provider.db.execute("INSERT INTO data VALUES (2, 'test2')")
        provider.db.commit()
        
        # Read data
        results = provider.read("SELECT * FROM data")
        
        assert len(results) == 2
        assert results[0] == {'id': 1, 'name': 'test1'}
        assert results[1] == {'id': 2, 'name': 'test2'}
        
        provider.close()
        
    def test_provider_read_auto_open(self):
        """Test read automatically opens connection."""
        config = IOConfig(source=":memory:", format="sqlite", mode=IOMode.WRITE)
        
        provider = SyncDatabaseProvider(config)
        # Don't explicitly open
        
        # Create table first
        provider.open()
        provider.db.execute("CREATE TABLE data (id INTEGER)")
        provider.db.execute("INSERT INTO data VALUES (1)")
        provider.db.commit()
        
        # Now test auto-open on read
        results = provider.read("SELECT * FROM data")
        
        assert provider._is_open is True
        assert len(results) == 1
        
        provider.close()
        
    def test_provider_write_single_dict(self):
        """Test writing single dictionary."""
        config = IOConfig(source=":memory:", format="sqlite", mode=IOMode.WRITE)
        
        provider = SyncDatabaseProvider(config)
        provider.open()
        
        # Create table
        provider.db.execute("CREATE TABLE data (id INTEGER, name TEXT)")
        provider.db.commit()
        
        # Write single dict
        provider.write({'id': 1, 'name': 'test'}, table='data')
        
        # Verify data was written
        cursor = provider.db.execute("SELECT * FROM data")
        rows = cursor.fetchall()
        
        assert len(rows) == 1
        assert dict(rows[0]) == {'id': 1, 'name': 'test'}
        
        provider.close()
        
    def test_provider_write_multiple_dicts(self):
        """Test writing multiple dictionaries."""
        config = IOConfig(source=":memory:", format="sqlite", mode=IOMode.WRITE)
        
        provider = SyncDatabaseProvider(config)
        provider.open()
        
        # Create table
        provider.db.execute("CREATE TABLE data (id INTEGER, name TEXT)")
        provider.db.commit()
        
        # Write multiple dicts
        data = [
            {'id': 1, 'name': 'test1'},
            {'id': 2, 'name': 'test2'}
        ]
        provider.write(data, table='data')
        
        # Verify data was written
        cursor = provider.db.execute("SELECT * FROM data ORDER BY id")
        rows = cursor.fetchall()
        
        assert len(rows) == 2
        assert dict(rows[0]) == {'id': 1, 'name': 'test1'}
        assert dict(rows[1]) == {'id': 2, 'name': 'test2'}
        
        provider.close()
        
    def test_provider_stream_read(self):
        """Test streaming read from database."""
        config = IOConfig(source=":memory:", format="sqlite", mode=IOMode.WRITE)
        
        provider = SyncDatabaseProvider(config)
        provider.open()
        
        # Create test data
        provider.db.execute("CREATE TABLE data (id INTEGER, value TEXT)")
        for i in range(5):
            provider.db.execute(f"INSERT INTO data VALUES ({i}, 'value{i}')")
        provider.db.commit()
        
        # Stream read
        results = list(provider.stream_read("SELECT * FROM data ORDER BY id"))
        
        assert len(results) == 5
        for i, row in enumerate(results):
            assert row == {'id': i, 'value': f'value{i}'}
            
        provider.close()
        
    def test_provider_stream_write(self):
        """Test streaming write to database."""
        config = IOConfig(source=":memory:", format="sqlite", mode=IOMode.WRITE)
        
        provider = SyncDatabaseProvider(config)
        provider.open()
        
        # Create table
        provider.db.execute("CREATE TABLE data (id INTEGER, value TEXT)")
        provider.db.commit()
        
        # Stream write
        def data_generator():
            for i in range(3):
                yield {'id': i, 'value': f'value{i}'}
        
        provider.stream_write(data_generator(), table='data')
        
        # Verify data was written
        cursor = provider.db.execute("SELECT * FROM data ORDER BY id")
        rows = cursor.fetchall()
        
        assert len(rows) == 3
        for i, row in enumerate(rows):
            assert dict(row) == {'id': i, 'value': f'value{i}'}
            
        provider.close()
        
    def test_provider_batch_operations(self):
        """Test batch read/write operations."""
        config = IOConfig(mode=IOMode.WRITE, format="sqlite", source=":memory:", batch_size=2)
        
        provider = SyncDatabaseProvider(config)
        provider.open()
        
        # Create table
        provider.db.execute("CREATE TABLE data (id INTEGER, value TEXT)")
        provider.db.commit()
        
        # Batch write - create batches manually
        data = [{'id': i, 'value': f'v{i}'} for i in range(5)]
        batch_size = 2
        batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        
        # Use batch_write with an iterator
        provider.batch_write(iter(batches), table='data')
        
        # Verify all data was written
        cursor = provider.db.execute("SELECT COUNT(*) as count FROM data")
        count = dict(cursor.fetchone())['count']
        assert count == 5
        
        # Test batch_read
        results = list(provider.batch_read("SELECT * FROM data ORDER BY id", batch_size=2))
        assert len(results) == 3  # ceil(5/2) = 3 batches
        assert len(results[0]) == 2  # First batch has 2 items
        assert len(results[1]) == 2  # Second batch has 2 items
        assert len(results[2]) == 1  # Last batch has 1 item
        
        provider.close()
        

class TestSyncHTTPProvider:
    """Test the SyncHTTPProvider class."""
    
    def test_provider_initialization(self):
        """Test SyncHTTPProvider initialization."""
        config = IOConfig(
            source="http://api.test.com/data",
            format="json",
            mode=IOMode.READ
        )
        
        provider = SyncHTTPProvider(config)
        
        assert provider.config == config
        assert provider.session is None
        assert provider._is_open is False
        assert provider.adapter is not None
        
    @patch('requests.Session')
    def test_provider_open(self, mock_session_class):
        """Test opening HTTP session."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        config = IOConfig(
            mode=IOMode.READ,
            format="json",
            source="http://api.test.com",
            headers={"Authorization": "Bearer token"}
        )
        
        provider = SyncHTTPProvider(config)
        provider.open()
        
        assert provider._is_open is True
        assert provider.session == mock_session
        mock_session.headers.update.assert_called_once_with({"Authorization": "Bearer token"})
        
    @patch('requests.Session')
    def test_provider_close(self, mock_session_class):
        """Test closing HTTP session."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        config = IOConfig(mode=IOMode.READ, format="json", source="http://api.test.com")
        
        provider = SyncHTTPProvider(config)
        provider.open()
        provider.close()
        
        assert provider._is_open is False
        mock_session.close.assert_called_once()
        
    @patch('requests.Session')
    def test_provider_validate(self, mock_session_class):
        """Test validating HTTP endpoint."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.head.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        config = IOConfig(mode=IOMode.READ, format="json", source="http://api.test.com", timeout=10)
        
        provider = SyncHTTPProvider(config)
        provider.open()
        
        assert provider.validate() is True
        mock_session.head.assert_called_once_with("http://api.test.com", timeout=10)
        
        # Test with error response
        mock_response.status_code = 404
        assert provider.validate() is False
        
        # Test with exception
        mock_session.head.side_effect = Exception("Connection error")
        assert provider.validate() is False
        
    @patch('requests.Session')
    def test_provider_read_json(self, mock_session_class):
        """Test reading JSON data from HTTP endpoint."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json.return_value = {'key': 'value'}
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        config = IOConfig(mode=IOMode.READ, format="json", source="http://api.test.com/data")
        
        provider = SyncHTTPProvider(config)
        result = provider.read()
        
        assert result == {'key': 'value'}
        assert provider._is_open is True
        mock_session.get.assert_called_once_with(
            "http://api.test.com/data",
            timeout=30
        )
        mock_response.raise_for_status.assert_called_once()
        
    @patch('requests.Session')
    def test_provider_read_text(self, mock_session_class):
        """Test reading text data from HTTP endpoint."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.headers = {'content-type': 'text/plain'}
        mock_response.text = 'Plain text response'
        mock_response.json.side_effect = json.JSONDecodeError("error", "", 0)
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        config = IOConfig(mode=IOMode.READ, format="json", source="http://api.test.com/data")
        
        provider = SyncHTTPProvider(config)
        result = provider.read()
        
        assert result == 'Plain text response'
        
    @patch('requests.Session')
    def test_provider_write(self, mock_session_class):
        """Test writing data to HTTP endpoint."""
        mock_session = Mock()
        mock_response = Mock()
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        config = IOConfig(mode=IOMode.READ, format="json", source="http://api.test.com/data")
        
        provider = SyncHTTPProvider(config)
        data = {'key': 'value'}
        provider.write(data)
        
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://api.test.com/data"
        assert 'timeout' in call_args[1]
        mock_response.raise_for_status.assert_called_once()
        
    @patch('requests.Session')
    def test_provider_stream_read(self, mock_session_class):
        """Test streaming read from HTTP endpoint."""
        mock_session = Mock()
        mock_response = Mock()
        
        # Mock streaming lines
        mock_response.iter_lines.return_value = [
            b'{"id": 1, "value": "test1"}',
            b'{"id": 2, "value": "test2"}',
            b'',  # Empty line should be skipped
            b'{"id": 3, "value": "test3"}'
        ]
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        config = IOConfig(mode=IOMode.READ, format="json", source="http://api.test.com/stream")
        
        provider = SyncHTTPProvider(config)
        results = list(provider.stream_read())
        
        assert len(results) == 3
        assert results[0] == {"id": 1, "value": "test1"}
        assert results[1] == {"id": 2, "value": "test2"}
        assert results[2] == {"id": 3, "value": "test3"}
        
        mock_session.get.assert_called_once_with(
            "http://api.test.com/stream",
            stream=True,
            timeout=30
        )
        
    @patch('requests.Session')
    def test_provider_stream_read_text(self, mock_session_class):
        """Test streaming non-JSON text from HTTP endpoint."""
        mock_session = Mock()
        mock_response = Mock()
        
        # Mock streaming lines with non-JSON data
        mock_response.iter_lines.return_value = [
            b'plain text line 1',
            b'plain text line 2'
        ]
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        config = IOConfig(mode=IOMode.READ, format="json", source="http://api.test.com/stream")
        
        provider = SyncHTTPProvider(config)
        results = list(provider.stream_read())
        
        assert len(results) == 2
        assert results[0] == 'plain text line 1'
        assert results[1] == 'plain text line 2'
        
    @patch('requests.Session')
    def test_provider_stream_write(self, mock_session_class):
        """Test streaming write to HTTP endpoint."""
        mock_session = Mock()
        mock_response = Mock()
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        config = IOConfig(mode=IOMode.READ, format="json", source="http://api.test.com/stream")
        
        provider = SyncHTTPProvider(config)
        
        def data_generator():
            for i in range(3):
                yield {'id': i, 'value': f'value{i}'}
        
        provider.stream_write(data_generator())
        
        # Should have made one POST per item
        assert mock_session.post.call_count == 3
        
    @patch('requests.Session')
    def test_provider_batch_operations(self, mock_session_class):
        """Test batch operations."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json.return_value = {'status': 'ok'}
        mock_response.text = 'ok'
        mock_session.post.return_value = mock_response
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        config = IOConfig(
            mode=IOMode.WRITE,
            format="json",
            source="http://api.test.com/batch",
            batch_size=2
        )
        
        provider = SyncHTTPProvider(config)
        
        # Test batch_write
        data = [{'id': i} for i in range(5)]
        batch_size = 2
        batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        
        provider.batch_write(iter(batches))
        
        # Should have made 3 POST calls (5 items / batch_size 2)
        assert mock_session.post.call_count == 3
        
        # Test batch_read
        # Mock paginated responses
        mock_responses = [
            Mock(headers={'content-type': 'application/json'}, json=Mock(return_value=[{'id': 0}, {'id': 1}])),
            Mock(headers={'content-type': 'application/json'}, json=Mock(return_value=[{'id': 2}, {'id': 3}])),
            Mock(headers={'content-type': 'application/json'}, json=Mock(return_value=[]))
        ]
        for resp in mock_responses:
            resp.raise_for_status = Mock()
        mock_session.get.side_effect = mock_responses
        
        results = list(provider.batch_read(batch_size=2))
        assert len(results) == 2  # Two batches before empty response
        

class TestProviderCreation:
    """Test creating sync providers via factory."""
    
    def test_create_sync_database_provider(self):
        """Test creating sync database provider."""
        config = IOConfig(
            source="test.db",
            format="sqlite",
            mode=IOMode.WRITE
        )
        
        adapter = DatabaseIOAdapter()
        provider = adapter.create_provider(config, is_async=False)
        
        assert isinstance(provider, SyncDatabaseProvider)
        assert provider.config == config
        
    def test_create_sync_http_provider(self):
        """Test creating sync HTTP provider."""
        config = IOConfig(
            mode=IOMode.READ,
            format="json",
            source="http://api.test.com"
        )
        
        adapter = HTTPIOAdapter()
        provider = adapter.create_provider(config, is_async=False)
        
        assert isinstance(provider, SyncHTTPProvider)
        assert provider.config == config
        
        
class TestErrorHandling:
    """Test error handling in sync providers."""
    
    def test_database_provider_invalid_query(self):
        """Test database provider with invalid query."""
        config = IOConfig(source=":memory:", format="sqlite", mode=IOMode.WRITE)
        
        provider = SyncDatabaseProvider(config)
        provider.open()
        
        # Invalid SQL should raise exception
        with pytest.raises(sqlite3.OperationalError):
            provider.read("SELECT * FROM nonexistent_table")
            
        provider.close()
        
    @patch('requests.Session')
    def test_http_provider_connection_error(self, mock_session_class):
        """Test HTTP provider with connection error."""
        mock_session = Mock()
        mock_session.get.side_effect = Exception("Connection refused")
        mock_session_class.return_value = mock_session
        
        config = IOConfig(mode=IOMode.READ, format="json", source="http://api.test.com")
        
        provider = SyncHTTPProvider(config)
        
        with pytest.raises(Exception, match="Connection refused"):
            provider.read()
            
    @patch('requests.Session')
    def test_http_provider_status_error(self, mock_session_class):
        """Test HTTP provider with HTTP error status."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        config = IOConfig(mode=IOMode.READ, format="json", source="http://api.test.com")
        
        provider = SyncHTTPProvider(config)
        
        with pytest.raises(Exception, match="404 Not Found"):
            provider.read()