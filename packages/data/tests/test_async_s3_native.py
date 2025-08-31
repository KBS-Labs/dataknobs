"""Tests for native async S3 backend with aioboto3 and session pooling."""

import asyncio
import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime

from dataknobs_data.backends.s3_async import AsyncS3Database
from dataknobs_data.records import Record
from dataknobs_data.query import Query, Operator, SortOrder


@pytest_asyncio.fixture
async def mock_s3_client():
    """Create a mock S3 client."""
    client = AsyncMock()
    
    # Mock basic operations
    client.put_object = AsyncMock()
    client.get_object = AsyncMock()
    client.delete_object = AsyncMock()
    client.head_object = AsyncMock()
    client.delete_objects = AsyncMock()
    
    # Mock paginator
    paginator = AsyncMock()
    
    async def mock_paginate(**kwargs):
        """Mock paginate to yield pages."""
        yield {
            'Contents': [
                {'Key': 'prefix/id1.json'},
                {'Key': 'prefix/id2.json'}
            ]
        }
    
    paginator.paginate = mock_paginate
    client.get_paginator = MagicMock(return_value=paginator)
    
    # Context manager support
    async def async_context_manager():
        return client
    
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock()
    
    return client


@pytest_asyncio.fixture
async def mock_session():
    """Create a mock aioboto3 session."""
    session = AsyncMock()
    
    # Create a mock client that will be returned from context manager
    mock_client_instance = AsyncMock()
    mock_client_instance.put_object = AsyncMock()
    mock_client_instance.get_object = AsyncMock()
    mock_client_instance.delete_object = AsyncMock()
    mock_client_instance.head_object = AsyncMock()
    mock_client_instance.delete_objects = AsyncMock()
    
    # Mock paginator
    paginator = AsyncMock()
    
    async def mock_paginate(**kwargs):
        yield {
            'Contents': [
                {'Key': 'test-prefix/id1.json'},
                {'Key': 'test-prefix/id2.json'}
            ]
        }
    
    paginator.paginate = mock_paginate
    mock_client_instance.get_paginator = MagicMock(return_value=paginator)
    
    # Create mock client factory that returns a context manager
    def client_factory(*args, **kwargs):
        mock_client_cm = AsyncMock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_cm.__aexit__ = AsyncMock()
        return mock_client_cm
    
    session.client = client_factory
    session.mock_client_instance = mock_client_instance  # Store reference for tests
    return session


@pytest_asyncio.fixture
async def s3_db():
    """Create an S3 database instance."""
    config = {
        "bucket": "test-bucket",
        "prefix": "test-prefix",
        "region": "us-east-1"
    }
    db = AsyncS3Database(config)
    return db


@pytest.mark.asyncio
async def test_connect_creates_session(s3_db, mock_session):
    """Test that connect creates and validates S3 session."""
    with patch('dataknobs_data.backends.s3_async.create_aioboto3_session',
               new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_session
        
        await s3_db.connect()
        
        assert s3_db._connected is True
        assert s3_db._session is not None


@pytest.mark.asyncio
async def test_close_releases_session(s3_db):
    """Test that close properly releases the session."""
    s3_db._session = mock_session
    s3_db._connected = True
    
    await s3_db.close()
    
    assert s3_db._connected is False
    assert s3_db._session is None


@pytest.mark.asyncio
async def test_create_record(s3_db, mock_session):
    """Test creating a record."""
    s3_db._session = mock_session
    s3_db._connected = True
    
    record = Record(data={"name": "test", "value": 42})
    
    result = await s3_db.create(record)
    
    assert result is not None  # Should return a UUID
    mock_session.mock_client_instance.put_object.assert_called_once()
    
    call_args = mock_session.mock_client_instance.put_object.call_args
    assert call_args.kwargs['Bucket'] == "test-bucket"
    assert call_args.kwargs['ContentType'] == "application/json"
    
    # Verify the body contains the record data
    body = json.loads(call_args.kwargs['Body'])
    assert body['fields']['name']['value'] == "test"
    assert body['fields']['value']['value'] == 42


@pytest.mark.asyncio
async def test_read_record(s3_db, mock_session):
    """Test reading a record."""
    s3_db._session = mock_session
    s3_db._connected = True
    
    # Setup mock response
    mock_client = mock_session.mock_client_instance
    mock_body = AsyncMock()
    mock_body.read = AsyncMock(return_value=json.dumps({
        "fields": {"name": {"value": "test"}},
        "metadata": {"id": "test-id"}
    }).encode())
    
    mock_client.get_object.return_value = {
        'Body': mock_body
    }
    
    result = await s3_db.read("test-id")
    
    assert result is not None
    assert result.get_field("name").value == "test"
    mock_client.get_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="test-prefix/test-id.json"
    )


@pytest.mark.asyncio
async def test_read_record_not_found(s3_db, mock_session):
    """Test reading a non-existent record."""
    s3_db._session = mock_session
    s3_db._connected = True
    
    mock_client = mock_session.mock_client_instance
    mock_client.get_object.side_effect = Exception("NoSuchKey")
    
    result = await s3_db.read("missing-id")
    
    assert result is None


@pytest.mark.asyncio
async def test_update_record(s3_db, mock_session):
    """Test updating a record."""
    s3_db._session = mock_session
    s3_db._connected = True
    
    # Mock exists check
    mock_client = mock_session.mock_client_instance
    mock_client.head_object.return_value = {}  # Exists
    
    record = Record(data={"name": "updated"})
    
    result = await s3_db.update("test-id", record)
    
    assert result is True
    mock_client.put_object.assert_called_once()
    
    call_args = mock_client.put_object.call_args
    assert call_args.kwargs['Bucket'] == "test-bucket"
    assert call_args.kwargs['Key'] == "test-prefix/test-id.json"


@pytest.mark.asyncio
async def test_delete_record(s3_db, mock_session):
    """Test deleting a record."""
    s3_db._session = mock_session
    s3_db._connected = True
    
    mock_client = mock_session.mock_client_instance
    
    result = await s3_db.delete("test-id")
    
    assert result is True
    mock_client.delete_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="test-prefix/test-id.json"
    )


@pytest.mark.asyncio
async def test_exists_record(s3_db, mock_session):
    """Test checking if record exists."""
    s3_db._session = mock_session
    s3_db._connected = True
    
    mock_client = mock_session.mock_client_instance
    mock_client.head_object.return_value = {}  # Exists
    
    result = await s3_db.exists("test-id")
    
    assert result is True
    mock_client.head_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="test-prefix/test-id.json"
    )


@pytest.mark.asyncio
async def test_search_with_filters(s3_db, mock_session):
    """Test searching with filters."""
    s3_db._session = mock_session
    s3_db._connected = True
    
    # Setup mock client and responses
    mock_client = mock_session.mock_client_instance
    
    # Mock paginator to return file list
    paginator = AsyncMock()
    
    async def mock_paginate(**kwargs):
        yield {
            'Contents': [
                {'Key': 'test-prefix/id1.json'},
                {'Key': 'test-prefix/id2.json'}
            ]
        }
    
    paginator.paginate = mock_paginate
    mock_client.get_paginator.return_value = paginator
    
    # Mock get_object responses
    call_count = 0
    
    async def mock_get_object(**kwargs):
        nonlocal call_count
        call_count += 1
        mock_body = AsyncMock()
        if call_count == 1:
            mock_body.read = AsyncMock(return_value=json.dumps({
                "fields": {"name": {"value": "doc1"}, "value": {"value": 10}},
                "metadata": {}
            }).encode())
        else:
            mock_body.read = AsyncMock(return_value=json.dumps({
                "fields": {"name": {"value": "doc2"}, "value": {"value": 20}},
                "metadata": {}
            }).encode())
        return {'Body': mock_body}
    
    mock_client.get_object = mock_get_object
    
    query = Query().filter("value", Operator.GT, 15).limit(10)
    
    results = await s3_db.search(query)
    
    assert len(results) == 1
    assert results[0].get_field("name").value == "doc2"


@pytest.mark.asyncio
async def test_search_with_sorting(s3_db, mock_session):
    """Test searching with sorting."""
    s3_db._session = mock_session
    s3_db._connected = True
    
    # Setup mock client
    mock_client = mock_session.mock_client_instance
    
    # Mock paginator
    paginator = AsyncMock()
    
    async def mock_paginate(**kwargs):
        yield {
            'Contents': [
                {'Key': 'test-prefix/id1.json'},
                {'Key': 'test-prefix/id2.json'},
                {'Key': 'test-prefix/id3.json'}
            ]
        }
    
    paginator.paginate = mock_paginate
    mock_client.get_paginator.return_value = paginator
    
    # Mock get_object responses
    responses = [
        {"fields": {"name": {"value": "C"}, "value": {"value": 30}}, "metadata": {}},
        {"fields": {"name": {"value": "A"}, "value": {"value": 10}}, "metadata": {}},
        {"fields": {"name": {"value": "B"}, "value": {"value": 20}}, "metadata": {}}
    ]
    
    call_count = 0
    
    async def mock_get_object(**kwargs):
        nonlocal call_count
        mock_body = AsyncMock()
        mock_body.read = AsyncMock(return_value=json.dumps(responses[call_count]).encode())
        call_count += 1
        return {'Body': mock_body}
    
    mock_client.get_object = mock_get_object
    
    query = Query().sort("name", SortOrder.ASC)
    
    results = await s3_db.search(query)
    
    assert len(results) == 3
    assert results[0].get_field("name").value == "A"
    assert results[1].get_field("name").value == "B"
    assert results[2].get_field("name").value == "C"


@pytest.mark.asyncio
async def test_count_all(s3_db, mock_session):
    """Test counting all records."""
    s3_db._session = mock_session
    s3_db._connected = True
    
    mock_client = mock_session.mock_client_instance
    
    count = await s3_db._count_all()
    
    assert count == 2  # Based on mock paginator returning 2 items


@pytest.mark.asyncio
async def test_clear_all(s3_db, mock_session):
    """Test clearing all records."""
    s3_db._session = mock_session
    s3_db._connected = True
    
    mock_client = mock_session.mock_client_instance
    mock_client.delete_objects.return_value = {}
    
    count = await s3_db.clear()
    
    assert count == 2
    mock_client.delete_objects.assert_called_once()
    
    call_args = mock_client.delete_objects.call_args
    assert call_args.kwargs['Bucket'] == "test-bucket"
    assert len(call_args.kwargs['Delete']['Objects']) == 2


@pytest.mark.asyncio
async def test_stream_read(s3_db, mock_session):
    """Test streaming read."""
    s3_db._session = mock_session
    s3_db._connected = True
    
    mock_client = mock_session.mock_client_instance
    
    # Mock get_object responses
    call_count = 0
    
    async def mock_get_object(**kwargs):
        nonlocal call_count
        call_count += 1
        mock_body = AsyncMock()
        mock_body.read = AsyncMock(return_value=json.dumps({
            "fields": {"name": {"value": f"doc{call_count}"}},
            "metadata": {}
        }).encode())
        return {'Body': mock_body}
    
    mock_client.get_object = mock_get_object
    
    records = []
    async for record in s3_db.stream_read():
        records.append(record)
    
    assert len(records) == 2
    assert records[0].get_field("name").value == "doc1"
    assert records[1].get_field("name").value == "doc2"


@pytest.mark.asyncio
async def test_stream_write(s3_db, mock_session):
    """Test streaming write."""
    s3_db._session = mock_session
    s3_db._connected = True
    
    mock_client = mock_session.mock_client_instance
    
    async def generate_records():
        for i in range(5):
            yield Record(data={"name": f"doc{i}"})
    
    result = await s3_db.stream_write(generate_records())
    
    assert result.successful == 5
    assert result.failed == 0
    assert result.total_processed == 5
    
    # Should have called put_object multiple times
    assert mock_client.put_object.call_count == 5


@pytest.mark.asyncio
async def test_stream_write_batch(s3_db, mock_session):
    """Test streaming write with batching."""
    s3_db._session = mock_session
    s3_db._connected = True
    
    mock_client = mock_session.mock_client_instance
    
    async def generate_records():
        for i in range(10):
            yield Record(data={"name": f"doc{i}"})
    
    from dataknobs_data.streaming import StreamConfig
    config = StreamConfig(batch_size=3)
    
    result = await s3_db.stream_write(generate_records(), config)
    
    assert result.successful == 10
    assert result.failed == 0
    assert result.total_processed == 10


@pytest.mark.asyncio
async def test_connection_pooling():
    """Test that session pooling works across event loops."""
    config = {
        "bucket": "test-bucket",
        "prefix": "test-prefix",
        "region": "us-east-1"
    }
    
    with patch('dataknobs_data.backends.s3_async._session_manager') as mock_manager:
        mock_session = AsyncMock()
        mock_manager.get_pool = AsyncMock(return_value=mock_session)
        
        # Create two databases
        db1 = AsyncS3Database(config)
        db2 = AsyncS3Database(config)
        
        # Connect both
        await db1.connect()
        await db2.connect()
        
        # Should use the same pool for same config
        assert mock_manager.get_pool.call_count == 2
        
        # Verify pool config is passed correctly
        calls = mock_manager.get_pool.call_args_list
        assert calls[0][0][0] == calls[1][0][0]  # Same config


@pytest.mark.asyncio
async def test_error_without_connection(s3_db):
    """Test that operations fail without connection."""
    record = Record(data={"test": "data"})
    
    with pytest.raises(RuntimeError, match="not connected"):
        await s3_db.create(record)
    
    with pytest.raises(RuntimeError, match="not connected"):
        await s3_db.read("test-id")
    
    with pytest.raises(RuntimeError, match="not connected"):
        await s3_db.search(Query())