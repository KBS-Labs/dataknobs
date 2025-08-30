#!/usr/bin/env python3
"""
Text-to-Vector Synchronization Example

This example demonstrates:
1. Automatic synchronization between text fields and vector embeddings
2. Change tracking to identify outdated vectors
3. Bulk synchronization of existing records
4. Real-time synchronization on updates

Requirements:
    pip install dataknobs-data sentence-transformers
"""

import asyncio
import time
from typing import List, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer

from dataknobs_data import AsyncDatabaseFactory, Record, VectorField, Query
from dataknobs_data.vector import VectorTextSynchronizer, SyncConfig


# Initialize embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a text string."""
    embedding = model.encode(text)
    return embedding.tolist()


class DocumentSync:
    """Helper class for document synchronization demo."""
    
    def __init__(self, db):
        self.db = db
        # Use the new simplified API
        self.synchronizer = VectorTextSynchronizer(
            database=db,
            embedding_fn=generate_embedding,
            text_fields=["title", "content"],  # Primary configuration
            vector_field="embedding",  # Sensible default
            field_separator=" ",
            auto_sync=True  # Enable auto-sync
        )
        print("✓ Synchronization configured")
    
    async def show_sync_status(self):
        """Display current synchronization status."""
        # For simplicity, we'll check which records don't have embeddings
        all_records = await self.db.all()
        outdated = [r for r in all_records if "embedding" not in r.data or r.data["embedding"] is None]
        total = len(all_records)
        
        print(f"\nSync Status:")
        print(f"  Total records: {total}")
        print(f"  Without embeddings: {len(outdated)}")
        print(f"  With embeddings: {total - len(outdated)}")
        
        return outdated


async def main():
    """Run the text-to-vector synchronization example."""
    
    # 1. Setup database
    print("\n1. Setting up database...")
    
    factory = AsyncDatabaseFactory()
    db = factory.create(
        backend="sqlite",
        database=":memory:",
        vector_enabled=True,
        vector_metric="cosine"
    )
    
    await db.connect()
    
    # 2. Create initial documents WITHOUT embeddings
    print("\n2. Creating documents without embeddings...")
    
    documents = [
        {
            "title": "Getting Started with Python",
            "content": "Python is a versatile programming language perfect for beginners.",
            "author": "Alice Smith",
            "created_at": datetime.now().isoformat()
        },
        {
            "title": "Advanced Python Techniques",
            "content": "Explore decorators, generators, and context managers in Python.",
            "author": "Bob Johnson",
            "created_at": datetime.now().isoformat()
        },
        {
            "title": "Data Science with Python",
            "content": "Learn to analyze data using pandas, numpy, and scikit-learn.",
            "author": "Carol White",
            "created_at": datetime.now().isoformat()
        },
        {
            "title": "Web Development Basics",
            "content": "Build modern web applications using HTML, CSS, and JavaScript.",
            "author": "David Brown",
            "created_at": datetime.now().isoformat()
        },
        {
            "title": "Database Design Principles",
            "content": "Understanding normalization, indexing, and query optimization.",
            "author": "Eve Davis",
            "created_at": datetime.now().isoformat()
        }
    ]
    
    # Create records without embeddings (simulating legacy data)
    record_ids = []
    for doc in documents:
        record = Record(doc)  # No embedding field
        record_id = await db.create(record)
        record_ids.append(record_id)
    
    print(f"✓ Created {len(record_ids)} documents without embeddings")
    
    # 3. Setup synchronization
    print("\n3. Setting up text-to-vector synchronization...")
    
    sync = DocumentSync(db)
    
    # Show initial status
    outdated = await sync.show_sync_status()
    
    # 4. Bulk synchronization
    print("\n4. Running bulk synchronization...")
    
    start_time = time.time()
    
    # Using the new sync_all method
    results = await sync.synchronizer.sync_all(
        force=True,  # Force sync even if vectors exist
        progress_callback=lambda status: print(f"  Progress: {status.processed_records}/{status.total_records} records processed")
    )
    
    elapsed = time.time() - start_time
    print(f"✓ Bulk sync completed in {elapsed:.2f} seconds")
    
    # Verify all records now have embeddings
    await sync.show_sync_status()
    
    # 5. Test vector search on synchronized data
    print("\n5. Testing vector search on synchronized data...")
    
    query_text = "Python programming for data analysis"
    query_embedding = generate_embedding(query_text)
    
    results = await db.vector_search(
        query_vector=query_embedding,
        k=3,
        vector_field="embedding"
    )
    
    print(f"Query: '{query_text}'")
    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.record['title']} (Score: {result.score:.3f})")
    
    # 6. Update a document and track changes
    print("\n6. Updating document and tracking changes...")
    
    # Update a document's text
    first_id = record_ids[0]
    await db.update(first_id, {
        "title": "Getting Started with Python 3.12",
        "content": "Python 3.12 brings exciting new features including improved error messages and performance."
    })
    
    print("✓ Updated document title and content")
    
    # Check sync status
    outdated = await sync.show_sync_status()
    
    if outdated:
        print(f"\nOutdated record detected: {outdated[0]['title']}")
    
    # 7. Incremental synchronization
    print("\n7. Running incremental sync for updated records...")
    
    # Sync only the outdated records
    for record in outdated:
        text = f"{record['title']} {record['content']}"
        embedding = generate_embedding(text)
        
        await db.update(record['id'], {
            "embedding": VectorField(embedding)  # Simplified - dimensions auto-detected
        })
        print(f"  ✓ Synced: {record['title']}")
    
    # Verify sync status
    await sync.show_sync_status()
    
    # 8. Auto-sync demonstration
    print("\n8. Demonstrating auto-sync on updates...")
    
    # Auto-sync is already enabled in constructor
    print("✓ Auto-sync is enabled")
    
    # Create a new document
    new_doc = Record({
        "title": "Machine Learning Fundamentals",
        "content": "Understanding supervised and unsupervised learning algorithms.",
        "author": "Frank Miller",
        "created_at": datetime.now().isoformat()
    })
    
    # With auto-sync, embedding should be added automatically
    new_id = await db.create(new_doc)
    
    # Note: In a real implementation, auto-sync would use database triggers
    # or event listeners. For this example, we'll manually trigger it.
    new_record = await db.read(new_id)
    success, updated_fields = await sync.synchronizer.sync_record(new_record)
    
    # Verify the new record has an embedding
    record = await db.read(new_id)
    if "embedding" in record and record["embedding"]:
        print(f"✓ New document automatically received embedding")
        print(f"  Embedding dimensions: {len(record['embedding'])}")
    
    # 9. Batch update demonstration
    print("\n9. Batch updating multiple documents...")
    
    # Update multiple documents
    updates = [
        (record_ids[1], {"content": "Master advanced Python concepts including metaclasses and descriptors."}),
        (record_ids[2], {"content": "Professional data science with Python, R, and Julia."}),
        (record_ids[3], {"title": "Full-Stack Web Development"})
    ]
    
    for record_id, update_data in updates:
        await db.update(record_id, update_data)
    
    print(f"✓ Updated {len(updates)} documents")
    
    # Check what needs syncing
    outdated = await sync.show_sync_status()
    
    # Batch sync all outdated records
    if outdated:
        print(f"\n  Syncing {len(outdated)} outdated records...")
        
        for record in outdated:
            text = f"{record.get('title', '')} {record.get('content', '')}"
            embedding = generate_embedding(text)
            
            await db.update(record['id'], {
                "embedding": VectorField(embedding)  # Simplified
            })
        
        print(f"  ✓ Batch sync completed")
    
    # Final status
    await sync.show_sync_status()
    
    # 10. Performance metrics
    print("\n10. Synchronization Performance Metrics:")
    
    total_records = await db.count()
    
    # Simulate checking sync performance
    sync_times = []
    for _ in range(3):
        start = time.time()
        text = "Sample text for performance testing"
        embedding = generate_embedding(text)
        sync_times.append(time.time() - start)
    
    avg_time = sum(sync_times) / len(sync_times)
    
    print(f"  Average embedding generation time: {avg_time*1000:.2f}ms")
    print(f"  Estimated time for {total_records} records: {avg_time*total_records:.2f}s")
    print(f"  Throughput: {1/avg_time:.0f} records/second")
    
    # Cleanup
    await db.close()
    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())