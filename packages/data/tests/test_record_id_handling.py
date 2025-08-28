"""Test that record IDs are properly handled across all database operations."""

import asyncio
import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.records import Record
from dataknobs_data.query import Query


@pytest.mark.asyncio
async def test_async_search_preserves_ids():
    """Test that async search preserves record IDs."""
    db = AsyncMemoryDatabase()
    await db.connect()
    
    # Create records
    record1 = await db.create(Record(data={"name": "Alice", "age": 30}))
    record2 = await db.create(Record(data={"name": "Bob", "age": 25}))
    
    # Search all records
    results = await db.search(Query())
    assert len(results) == 2
    
    # Check that IDs are preserved
    ids = {r.id for r in results}
    assert record1 in ids
    assert record2 in ids
    
    # Check specific search
    results = await db.search(Query().filter("name", "==", "Alice"))
    assert len(results) == 1
    assert results[0].id == record1
    
    await db.close()


@pytest.mark.asyncio
async def test_async_all_preserves_ids():
    """Test that the all() method preserves record IDs."""
    db = AsyncMemoryDatabase()
    await db.connect()
    
    # Create records
    record1 = await db.create(Record(data={"name": "Alice"}))
    record2 = await db.create(Record(data={"name": "Bob"}))
    
    # Get all records
    results = await db.all()
    assert len(results) == 2
    
    # Check that IDs are preserved
    ids = {r.id for r in results}
    assert record1 in ids
    assert record2 in ids
    
    await db.close()


@pytest.mark.asyncio
async def test_async_stream_read_preserves_ids():
    """Test that stream_read preserves record IDs."""
    db = AsyncMemoryDatabase()
    await db.connect()
    
    # Create records
    record1 = await db.create(Record(data={"name": "Alice"}))
    record2 = await db.create(Record(data={"name": "Bob"}))
    
    # Stream all records without query
    streamed_ids = []
    async for record in db.stream_read():
        assert record.id is not None
        streamed_ids.append(record.id)
    
    assert record1 in streamed_ids
    assert record2 in streamed_ids
    
    # Stream with query
    streamed_ids = []
    async for record in db.stream_read(Query().filter("name", "==", "Alice")):
        assert record.id is not None
        streamed_ids.append(record.id)
    
    assert record1 in streamed_ids
    assert record2 not in streamed_ids
    
    await db.close()


def test_sync_search_preserves_ids():
    """Test that sync search preserves record IDs."""
    db = SyncMemoryDatabase()
    db.connect()
    
    # Create records
    record1 = db.create(Record(data={"name": "Alice", "age": 30}))
    record2 = db.create(Record(data={"name": "Bob", "age": 25}))
    
    # Search all records
    results = db.search(Query())
    assert len(results) == 2
    
    # Check that IDs are preserved
    ids = {r.id for r in results}
    assert record1 in ids
    assert record2 in ids
    
    # Check specific search
    results = db.search(Query().filter("name", "==", "Alice"))
    assert len(results) == 1
    assert results[0].id == record1
    
    db.close()


def test_sync_all_preserves_ids():
    """Test that the sync all() method preserves record IDs."""
    db = SyncMemoryDatabase()
    db.connect()
    
    # Create records
    record1 = db.create(Record(data={"name": "Alice"}))
    record2 = db.create(Record(data={"name": "Bob"}))
    
    # Get all records
    results = db.all()
    assert len(results) == 2
    
    # Check that IDs are preserved
    ids = {r.id for r in results}
    assert record1 in ids
    assert record2 in ids
    
    db.close()


def test_sync_stream_read_preserves_ids():
    """Test that sync stream_read preserves record IDs."""
    db = SyncMemoryDatabase()
    db.connect()
    
    # Create records
    record1 = db.create(Record(data={"name": "Alice"}))
    record2 = db.create(Record(data={"name": "Bob"}))
    
    # Stream all records without query
    streamed_ids = []
    for record in db.stream_read():
        assert record.id is not None
        streamed_ids.append(record.id)
    
    assert record1 in streamed_ids
    assert record2 in streamed_ids
    
    # Stream with query
    streamed_ids = []
    for record in db.stream_read(Query().filter("name", "==", "Alice")):
        assert record.id is not None
        streamed_ids.append(record.id)
    
    assert record1 in streamed_ids
    assert record2 not in streamed_ids
    
    db.close()