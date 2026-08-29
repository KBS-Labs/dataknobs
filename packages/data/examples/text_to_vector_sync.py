#!/usr/bin/env python3
"""Text-to-Vector Synchronization Example

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
from collections.abc import Callable
from datetime import datetime
from functools import cache

from dataknobs_data import AsyncDatabaseFactory, Record, VectorField
from dataknobs_data.vector import VectorTextSynchronizer


MODEL_NAME = "all-MiniLM-L6-v2"

EmbeddingFn = Callable[[str], list[float]]


@cache
def _embedding_model():
    """Load the sentence-transformers model once, on first use.

    Both the import and the construction are deferred. Loading a model
    downloads weights and takes seconds, so doing it as an import side
    effect makes this file impossible to import without the optional
    dependency installed -- and a file nothing can import is a file
    nothing can test.
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model ({MODEL_NAME})...")
    return SentenceTransformer(MODEL_NAME)


def generate_embedding(text: str) -> list[float]:
    """Generate an embedding for a text string using the default model."""
    return _embedding_model().encode(text).tolist()


class DocumentSync:
    """Helper class for document synchronization demo."""

    def __init__(self, db, embedding_fn: EmbeddingFn = generate_embedding):
        self.db = db
        # Use the new simplified API
        self.synchronizer = VectorTextSynchronizer(
            database=db,
            embedding_fn=embedding_fn,
            text_fields=["title", "content"],  # Primary configuration
            vector_field="embedding",  # Sensible default
            field_separator=" ",
            auto_sync=True,  # Enable auto-sync
        )
        print("✓ Synchronization configured")

    async def show_sync_status(self):
        """Display current synchronization status."""
        # For simplicity, we'll check which records don't have embeddings
        all_records = await self.db.all()
        outdated = [
            r for r in all_records if "embedding" not in r.data or r.data["embedding"] is None
        ]
        total = len(all_records)

        print("\nSync Status:")
        print(f"  Total records: {total}")
        print(f"  Without embeddings: {len(outdated)}")
        print(f"  With embeddings: {total - len(outdated)}")

        return outdated


async def main(embedding_fn: EmbeddingFn = generate_embedding):
    """Run the text-to-vector synchronization example.

    Args:
        embedding_fn: Text-to-vector function. Defaults to the
            sentence-transformers model, loaded on first call.
    """
    # 1. Setup database
    print("\n1. Setting up database...")

    factory = AsyncDatabaseFactory()
    db = factory.create(
        backend="sqlite", path=":memory:", vector_enabled=True, vector_metric="cosine"
    )

    await db.connect()

    # The database owns an aiosqlite connection thread. Closing it only on
    # the happy path -- as this used to -- means any failure below leaves
    # that thread running and hangs interpreter shutdown, turning a legible
    # traceback into a process that never exits.
    try:
        # 2. Create initial documents WITHOUT embeddings
        print("\n2. Creating documents without embeddings...")

        documents = [
            {
                "title": "Getting Started with Python",
                "content": "Python is a versatile programming language perfect for beginners.",
                "author": "Alice Smith",
                "created_at": datetime.now().isoformat(),
            },
            {
                "title": "Advanced Python Techniques",
                "content": "Explore decorators, generators, and context managers in Python.",
                "author": "Bob Johnson",
                "created_at": datetime.now().isoformat(),
            },
            {
                "title": "Data Science with Python",
                "content": "Learn to analyze data using pandas, numpy, and scikit-learn.",
                "author": "Carol White",
                "created_at": datetime.now().isoformat(),
            },
            {
                "title": "Web Development Basics",
                "content": "Build modern web applications using HTML, CSS, and JavaScript.",
                "author": "David Brown",
                "created_at": datetime.now().isoformat(),
            },
            {
                "title": "Database Design Principles",
                "content": "Understanding normalization, indexing, and query optimization.",
                "author": "Eve Davis",
                "created_at": datetime.now().isoformat(),
            },
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

        sync = DocumentSync(db, embedding_fn)

        # Show initial status
        outdated = await sync.show_sync_status()

        # 4. Bulk synchronization
        print("\n4. Running bulk synchronization...")

        start_time = time.time()

        # Using the new sync_all method
        results = await sync.synchronizer.sync_all(
            force=True,  # Force sync even if vectors exist
            # sync_all's callback contract is (done, total), not a status object.
            progress_callback=lambda done, total: print(
                f"  Progress: {done}/{total} records processed"
            ),
        )

        elapsed = time.time() - start_time
        print(f"✓ Bulk sync completed in {elapsed:.2f} seconds")

        # Verify all records now have embeddings
        await sync.show_sync_status()

        # 5. Test vector search on synchronized data
        print("\n5. Testing vector search on synchronized data...")

        query_text = "Python programming for data analysis"
        query_embedding = embedding_fn(query_text)

        results = await db.vector_search(
            query_vector=query_embedding, k=3, vector_field="embedding"
        )

        print(f"Query: '{query_text}'")
        print("Results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.record['title']} (Score: {result.score:.3f})")

        # 6. Update a document and track changes
        print("\n6. Updating document and tracking changes...")

        # Update a document's text
        # `update` takes a Record and replaces the stored one wholesale, so read
        # first and edit in place -- passing a bare dict of the changed fields
        # raises AttributeError, and would drop every field it omits if it did not.
        first_id = record_ids[0]
        first_record = await db.read(first_id)
        old_data = dict(first_record.data)
        first_record["title"] = "Getting Started with Python 3.12"
        first_record["content"] = (
            "Python 3.12 brings exciting new features including "
            "improved error messages and performance."
        )
        await db.update(first_id, first_record)

        print("✓ Updated document title and content")

        # Check sync status. Note what this does and does not tell you: it
        # reports records with *no* embedding. A record whose text just
        # changed still has one -- a stale one -- so it is absent from this
        # list, and step 7 below is what handles that case.
        missing = await sync.show_sync_status()

        # 7. Incremental synchronization
        print("\n7. Re-syncing the record whose text changed...")

        changed_fields = [k for k in ("title", "content") if old_data.get(k) != first_record[k]]
        print(f"  changed source fields: {changed_fields}")

        # `sync_on_update` is the change-tracking entry point -- give it the old
        # and new data and it re-embeds only what actually changed. Both source
        # fields did, so it does the work and reports True.
        synced = await sync.synchronizer.sync_on_update(first_id, old_data, dict(first_record.data))
        print(f"  sync_on_update -> {synced}")

        # `sync_record` is the whole-record entry point, and it is not
        # unconditional: it skips any vector field whose stored digest still
        # matches the text the record would produce now. The line above just
        # brought this record up to date, so there is nothing left to do and it
        # reports no updated fields. `force=True` is how you ask for an
        # embedding regardless -- after a model change, say.
        success, updated = await sync.synchronizer.sync_record(first_id)
        print(f"  sync_record -> success={success}, updated={updated}")

        # And anything still missing an embedding gets one.
        for record in missing:
            text = f"{record.get('title', '')} {record.get('content', '')}"
            # `record.id` is the storage id; `record["id"]` would look for a
            # field named "id", which these documents do not have. And `update`
            # replaces the whole record, so set the field on the record already
            # in hand rather than handing it a dict of just the change.
            record["embedding"] = VectorField(embedding_fn(text))
            await db.update(record.id, record)
            print(f"  ✓ Synced: {record['title']}")

        # Verify sync status
        await sync.show_sync_status()

        # 8. Auto-sync demonstration
        print("\n8. Demonstrating auto-sync on updates...")

        # Auto-sync is already enabled in constructor
        print("✓ Auto-sync is enabled")

        # Create a new document
        new_doc = Record(
            {
                "title": "Machine Learning Fundamentals",
                "content": "Understanding supervised and unsupervised learning algorithms.",
                "author": "Frank Miller",
                "created_at": datetime.now().isoformat(),
            }
        )

        # With auto-sync, embedding should be added automatically
        new_id = await db.create(new_doc)

        # Note: In a real implementation, auto-sync would use database triggers
        # or event listeners. For this example, we'll manually trigger it.
        new_record = await db.read(new_id)
        success, updated_fields = await sync.synchronizer.sync_record(new_record)
        print(f"  sync_record -> success={success}, updated fields={updated_fields}")

        # Verify the new record has an embedding
        record = await db.read(new_id)
        # Field presence, not truthiness: the value is a numpy array, and
        # `if array:` raises "truth value ... is ambiguous".
        if "embedding" in record:
            print("✓ New document automatically received embedding")
            print(f"  Embedding dimensions: {len(record['embedding'])}")

        # 9. Batch update demonstration
        print("\n9. Batch updating multiple documents...")

        # Update multiple documents
        updates = [
            (
                record_ids[1],
                {
                    "content": "Master advanced Python concepts including metaclasses and descriptors."
                },
            ),
            (record_ids[2], {"content": "Professional data science with Python, R, and Julia."}),
            (record_ids[3], {"title": "Full-Stack Web Development"}),
        ]

        for record_id, update_data in updates:
            existing = await db.read(record_id)
            for key, value in update_data.items():
                existing[key] = value
            await db.update(record_id, existing)

        print(f"✓ Updated {len(updates)} documents")

        # Check what needs syncing
        outdated = await sync.show_sync_status()

        # Batch sync all outdated records
        if outdated:
            print(f"\n  Syncing {len(outdated)} outdated records...")

            for record in outdated:
                text = f"{record.get('title', '')} {record.get('content', '')}"
                embedding = embedding_fn(text)

                record["embedding"] = VectorField(embedding)
                await db.update(record.id, record)

            print("  ✓ Batch sync completed")

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
            embedding = embedding_fn(text)
            sync_times.append(time.time() - start)

        avg_time = sum(sync_times) / len(sync_times)

        print(f"  Average embedding generation time: {avg_time * 1000:.2f}ms")
        print(f"  Estimated time for {total_records} records: {avg_time * total_records:.2f}s")
        print(f"  Throughput: {1 / avg_time:.0f} records/second")

        print("\n✓ Example completed successfully!")

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
