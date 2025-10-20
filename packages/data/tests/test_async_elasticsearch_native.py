"""Tests for native async Elasticsearch backend with connection pooling."""

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.records import Record
from dataknobs_data.query import Query, Operator


@pytest_asyncio.fixture
async def mock_es_client():
    """Create a mock Elasticsearch client."""
    client = AsyncMock()
    
    # Mock index operations
    client.indices.exists = AsyncMock(return_value=True)
    client.indices.create = AsyncMock()
    
    # Mock document operations
    client.index = AsyncMock(return_value={"_id": "test-id"})
    client.get = AsyncMock(return_value={
        "_id": "test-id",
        "_source": {
            "data": {"name": "test"},
            "metadata": {}
        }
    })
    client.update = AsyncMock()
    client.delete = AsyncMock()
    client.exists = AsyncMock(return_value=True)
    client.count = AsyncMock(return_value={"count": 5})
    client.delete_by_query = AsyncMock(return_value={"deleted": 5})
    client.search = AsyncMock(return_value={
        "hits": {
            "hits": [
                {
                    "_id": "1",
                    "_source": {
                        "data": {"name": "doc1"},
                        "metadata": {}
                    }
                }
            ]
        }
    })
    
    # Mock scroll operations
    client.scroll = AsyncMock(return_value={
        "_scroll_id": "scroll-123",
        "hits": {"hits": []}
    })
    client.clear_scroll = AsyncMock()
    
    # Mock bulk operations
    client.bulk = AsyncMock()
    
    # Mock close
    client.close = AsyncMock()
    
    return client


@pytest_asyncio.fixture
async def es_db():
    """Create an Elasticsearch database instance."""
    # Clear the global pool manager state to ensure fresh connections for each test
    from dataknobs_data.backends.elasticsearch_async import _client_manager
    await _client_manager.close_all()

    config = {
        "hosts": ["localhost:9200"],
        "index": "test_index"
    }
    db = AsyncElasticsearchDatabase(config)
    yield db

    # Cleanup after test
    if db._connected:
        await db.close()
    await _client_manager.close_all()


@pytest.mark.asyncio
async def test_connect_creates_client(es_db):
    """Test that connect creates and validates Elasticsearch client."""
    # Create a fresh mock client for this test
    mock_client = AsyncMock()
    mock_client.indices.exists = AsyncMock(return_value=True)
    mock_client.indices.create = AsyncMock()

    with patch('dataknobs_data.backends.elasticsearch_async.create_async_elasticsearch_client',
               new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_client

        await es_db.connect()

        assert es_db._connected is True
        assert es_db._client is not None
        mock_client.indices.exists.assert_called_once_with(index="test_index")


@pytest.mark.asyncio
async def test_connect_creates_index_if_missing(es_db):
    """Test that connect creates index if it doesn't exist."""
    # Create a fresh mock client with index not existing
    mock_client = AsyncMock()
    mock_client.indices.exists = AsyncMock(return_value=False)
    mock_client.indices.create = AsyncMock()

    with patch('dataknobs_data.backends.elasticsearch_async.create_async_elasticsearch_client',
               new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_client

        await es_db.connect()

        mock_client.indices.create.assert_called_once()
        call_args = mock_client.indices.create.call_args
        assert call_args.kwargs['index'] == "test_index"
        assert 'mappings' in call_args.kwargs


@pytest.mark.asyncio
async def test_close_releases_client(es_db, mock_es_client):
    """Test that close properly releases the client."""
    es_db._client = mock_es_client
    es_db._connected = True
    
    await es_db.close()
    
    assert es_db._connected is False
    assert es_db._client is None
    # Client is managed by pool manager, so close is not called directly
    mock_es_client.close.assert_not_called()


@pytest.mark.asyncio
async def test_create_record(es_db, mock_es_client):
    """Test creating a record."""
    es_db._client = mock_es_client
    es_db._connected = True
    
    record = Record(data={"name": "test", "value": 42})
    
    result = await es_db.create(record)
    
    assert result == "test-id"
    mock_es_client.index.assert_called_once()
    call_args = mock_es_client.index.call_args
    assert call_args.kwargs['index'] == "test_index"
    assert "data" in call_args.kwargs['document']
    assert call_args.kwargs['document']['data']['name'] == "test"


@pytest.mark.asyncio
async def test_read_record(es_db, mock_es_client):
    """Test reading a record."""
    es_db._client = mock_es_client
    es_db._connected = True
    
    result = await es_db.read("test-id")
    
    assert result is not None
    assert result.get_field("name").value == "test"
    mock_es_client.get.assert_called_once_with(
        index="test_index",
        id="test-id"
    )


@pytest.mark.asyncio
async def test_read_record_not_found(es_db, mock_es_client):
    """Test reading a non-existent record."""
    es_db._client = mock_es_client
    es_db._connected = True
    mock_es_client.get.side_effect = Exception("Not found")
    
    result = await es_db.read("missing-id")
    
    assert result is None


@pytest.mark.asyncio
async def test_update_record(es_db, mock_es_client):
    """Test updating a record."""
    es_db._client = mock_es_client
    es_db._connected = True
    
    record = Record(data={"name": "updated"})
    
    result = await es_db.update("test-id", record)
    
    assert result is True
    mock_es_client.update.assert_called_once()
    call_args = mock_es_client.update.call_args
    assert call_args.kwargs['index'] == "test_index"
    assert call_args.kwargs['id'] == "test-id"


@pytest.mark.asyncio
async def test_delete_record(es_db, mock_es_client):
    """Test deleting a record."""
    es_db._client = mock_es_client
    es_db._connected = True
    
    result = await es_db.delete("test-id")
    
    assert result is True
    mock_es_client.delete.assert_called_once_with(
        index="test_index",
        id="test-id",
        refresh=True
    )


@pytest.mark.asyncio
async def test_exists_record(es_db, mock_es_client):
    """Test checking if record exists."""
    es_db._client = mock_es_client
    es_db._connected = True
    
    result = await es_db.exists("test-id")
    
    assert result is True
    mock_es_client.exists.assert_called_once_with(
        index="test_index",
        id="test-id"
    )


@pytest.mark.asyncio
async def test_search_with_filters(es_db, mock_es_client):
    """Test searching with filters."""
    es_db._client = mock_es_client
    es_db._connected = True
    
    query = Query().filter("name", Operator.EQ, "doc1").limit(10)
    
    results = await es_db.search(query)
    
    assert len(results) == 1
    assert results[0].get_field("name").value == "doc1"
    
    mock_es_client.search.assert_called_once()
    call_args = mock_es_client.search.call_args
    assert call_args.kwargs['index'] == "test_index"
    assert 'query' in call_args.kwargs


@pytest.mark.asyncio
async def test_count_all(es_db, mock_es_client):
    """Test counting all records."""
    es_db._client = mock_es_client
    es_db._connected = True
    
    count = await es_db._count_all()
    
    assert count == 5
    mock_es_client.count.assert_called_once_with(index="test_index")


@pytest.mark.asyncio
async def test_clear_all(es_db, mock_es_client):
    """Test clearing all records."""
    es_db._client = mock_es_client
    es_db._connected = True
    
    count = await es_db.clear()
    
    assert count == 5
    mock_es_client.delete_by_query.assert_called_once()
    call_args = mock_es_client.delete_by_query.call_args
    assert call_args.kwargs['index'] == "test_index"
    assert call_args.kwargs['query']['match_all'] == {}


@pytest.mark.asyncio
async def test_stream_read(es_db, mock_es_client):
    """Test streaming read."""
    es_db._client = mock_es_client
    es_db._connected = True
    
    # Setup mock responses for scroll
    mock_es_client.search.return_value = {
        "_scroll_id": "scroll-123",
        "hits": {
            "hits": [
                {
                    "_id": "1",
                    "_source": {
                        "data": {"name": "doc1"},
                        "metadata": {}
                    }
                },
                {
                    "_id": "2", 
                    "_source": {
                        "data": {"name": "doc2"},
                        "metadata": {}
                    }
                }
            ]
        }
    }
    
    records = []
    async for record in es_db.stream_read():
        records.append(record)
    
    assert len(records) == 2
    assert records[0].get_field("name").value == "doc1"
    assert records[1].get_field("name").value == "doc2"
    
    mock_es_client.search.assert_called_once()
    mock_es_client.clear_scroll.assert_called_once_with(scroll_id="scroll-123")


@pytest.mark.asyncio
async def test_stream_write(es_db, mock_es_client):
    """Test streaming write."""
    es_db._client = mock_es_client
    es_db._connected = True
    
    async def generate_records():
        for i in range(5):
            yield Record(data={"name": f"doc{i}"})
    
    result = await es_db.stream_write(generate_records())
    
    assert result.successful == 5
    assert result.failed == 0
    assert result.total_processed == 5
    
    # Should have called bulk API
    mock_es_client.bulk.assert_called()


@pytest.mark.asyncio
async def test_connection_pooling():
    """Test that connection pooling works across event loops."""
    # Clean up any existing pool state first
    from dataknobs_data.backends.elasticsearch_async import _client_manager
    await _client_manager.close_all()

    config = {
        "hosts": ["localhost:9200"],
        "index": "test_index"
    }

    with patch('dataknobs_data.backends.elasticsearch_async._client_manager') as mock_manager:
        mock_pool = AsyncMock()
        mock_manager.get_pool = AsyncMock(return_value=mock_pool)
        mock_manager.close_all = AsyncMock()

        # Create two databases
        db1 = AsyncElasticsearchDatabase(config)
        db2 = AsyncElasticsearchDatabase(config)

        # Connect both
        await db1.connect()
        await db2.connect()

        # Should use the same pool for same config
        assert mock_manager.get_pool.call_count == 2

        # Verify pool config is passed correctly
        calls = mock_manager.get_pool.call_args_list
        assert calls[0][0][0] == calls[1][0][0]  # Same config

        # Cleanup
        await db1.close()
        await db2.close()

    # Cleanup real manager state after test
    await _client_manager.close_all()


@pytest.mark.asyncio
async def test_error_without_connection(es_db):
    """Test that operations fail without connection."""
    record = Record(data={"test": "data"})
    
    with pytest.raises(RuntimeError, match="not connected"):
        await es_db.create(record)
    
    with pytest.raises(RuntimeError, match="not connected"):
        await es_db.read("test-id")
    
    with pytest.raises(RuntimeError, match="not connected"):
        await es_db.search(Query())