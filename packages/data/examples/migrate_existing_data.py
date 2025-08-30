#!/usr/bin/env python3
"""
Migration Example - Adding Vectors to Existing Data

This example demonstrates:
1. Migrating a database without vectors to include vector support
2. Adding vector fields to existing records
3. Incremental vectorization for large datasets
4. Handling migration errors and retries
5. Verifying migration completeness

Requirements:
    pip install dataknobs-data sentence-transformers tqdm
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

from dataknobs_data import AsyncDatabaseFactory, Record, VectorField, Query
from dataknobs_data.vector import VectorMigration, IncrementalVectorizer


# Initialize embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a text string."""
    embedding = model.encode(text)
    return embedding.tolist()


@dataclass
class MigrationStats:
    """Track migration statistics."""
    total_records: int = 0
    migrated_records: int = 0
    failed_records: int = 0
    start_time: float = 0
    end_time: float = 0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time else time.time() - self.start_time
    
    @property
    def success_rate(self) -> float:
        return (self.migrated_records / self.total_records * 100) if self.total_records > 0 else 0


async def create_legacy_database():
    """Create a simulated legacy database without vector support."""
    
    print("\n1. Creating legacy database (without vectors)...")
    
    # Create database without vector support
    factory = AsyncDatabaseFactory()
    db = factory.create(
        backend="sqlite",
        database=":memory:",
        vector_enabled=False  # No vector support initially
    )
    
    await db.connect()
    
    # Create sample legacy data
    legacy_data = [
        {
            "id": 1,
            "type": "article",
            "title": "Introduction to Cloud Computing",
            "content": "Cloud computing revolutionizes how we store and process data using remote servers.",
            "author": "John Doe",
            "published": "2023-01-15",
            "tags": ["cloud", "technology", "infrastructure"]
        },
        {
            "id": 2,
            "type": "article",
            "title": "Microservices Architecture",
            "content": "Breaking down monolithic applications into smaller, independent services.",
            "author": "Jane Smith",
            "published": "2023-02-20",
            "tags": ["architecture", "microservices", "design"]
        },
        {
            "id": 3,
            "type": "tutorial",
            "title": "Docker Container Basics",
            "content": "Learn how to containerize applications using Docker for consistent deployments.",
            "author": "Bob Wilson",
            "published": "2023-03-10",
            "tags": ["docker", "containers", "devops"]
        },
        {
            "id": 4,
            "type": "tutorial",
            "title": "Kubernetes Orchestration",
            "content": "Managing containerized applications at scale with Kubernetes orchestration.",
            "author": "Alice Brown",
            "published": "2023-04-05",
            "tags": ["kubernetes", "orchestration", "containers"]
        },
        {
            "id": 5,
            "type": "guide",
            "title": "CI/CD Pipeline Setup",
            "content": "Implementing continuous integration and deployment pipelines for modern software.",
            "author": "Charlie Davis",
            "published": "2023-05-12",
            "tags": ["ci/cd", "automation", "devops"]
        },
        {
            "id": 6,
            "type": "article",
            "title": "Serverless Computing",
            "content": "Running applications without managing servers using Function-as-a-Service platforms.",
            "author": "Diana Miller",
            "published": "2023-06-18",
            "tags": ["serverless", "cloud", "functions"]
        },
        {
            "id": 7,
            "type": "tutorial",
            "title": "API Design Best Practices",
            "content": "Creating well-designed RESTful APIs that are scalable and maintainable.",
            "author": "Edward Jones",
            "published": "2023-07-22",
            "tags": ["api", "rest", "design"]
        },
        {
            "id": 8,
            "type": "guide",
            "title": "Database Optimization",
            "content": "Techniques for improving database performance through indexing and query optimization.",
            "author": "Fiona Taylor",
            "published": "2023-08-30",
            "tags": ["database", "optimization", "performance"]
        }
    ]
    
    # Insert legacy data
    for data in legacy_data:
        await db.create(Record(data))
    
    print(f"✓ Created legacy database with {len(legacy_data)} records (no vectors)")
    
    return db


async def migrate_to_vector_database(legacy_db):
    """Migrate legacy database to include vector support."""
    
    print("\n2. Creating new database with vector support...")
    
    # Create new database with vector support
    factory = AsyncDatabaseFactory()
    vector_db = factory.create(
        backend="sqlite",
        database=":memory:",
        vector_enabled=True,  # Enable vector support
        vector_metric="cosine"
    )
    
    await vector_db.connect()
    
    print("✓ Created vector-enabled database")
    
    return vector_db


async def manual_migration(legacy_db, vector_db, stats: MigrationStats):
    """Manually migrate records with vector generation."""
    
    print("\n3. Manual Migration Process...")
    
    # Get all records from legacy database
    all_records = await legacy_db.find()
    stats.total_records = len(all_records)
    
    print(f"  Found {stats.total_records} records to migrate")
    
    # Migrate in batches
    batch_size = 3
    failed_records = []
    
    for i in range(0, len(all_records), batch_size):
        batch = all_records[i:i + batch_size]
        print(f"\n  Processing batch {i//batch_size + 1} ({len(batch)} records)...")
        
        for record in batch:
            try:
                # Generate embedding from title and content
                text = f"{record.get('title', '')} {record.get('content', '')}"
                embedding = generate_embedding(text)
                
                # Create new record with embedding
                new_record = Record({
                    **record,
                    "embedding": VectorField(embedding),  # Simplified - dimensions auto-detected
                    "migrated_at": datetime.now().isoformat()
                })
                
                await vector_db.create(new_record)
                stats.migrated_records += 1
                
                print(f"    ✓ Migrated: {record.get('title', 'Unknown')[:50]}")
                
            except Exception as e:
                stats.failed_records += 1
                failed_records.append({
                    "record": record,
                    "error": str(e)
                })
                print(f"    ✗ Failed: {record.get('title', 'Unknown')[:50]} - {e}")
    
    return failed_records


async def incremental_migration(vector_db):
    """Demonstrate incremental vectorization for large datasets."""
    
    print("\n4. Incremental Vectorization (for large datasets)...")
    
    # Create incremental vectorizer with simplified API
    vectorizer = IncrementalVectorizer(
        database=vector_db,
        embedding_fn=generate_embedding,
        text_fields=["title", "content"],  # Primary configuration
        vector_field="embedding_v2",  # Additional embedding field
        batch_size=2,
        checkpoint_interval=5  # Save progress every 5 records
    )
    
    # Start vectorization with progress tracking
    print("  Starting incremental vectorization...")
    
    async def progress_callback(completed: int, total: int, current_batch: List[str]):
        percentage = (completed / total * 100) if total > 0 else 0
        print(f"    Progress: {completed}/{total} ({percentage:.1f}%) - Processing: {current_batch[0][:30]}...")
    
    # Run vectorization with simplified API
    results = await vectorizer.run()
    
    print(f"  ✓ Incremental vectorization completed")
    print(f"    Processed: {results.processed} records")
    print(f"    Failed: {results.failed} records")
    
    return results


async def verify_migration(vector_db):
    """Verify that migration was successful."""
    
    print("\n5. Verifying Migration...")
    
    # Check total records
    total_records = await vector_db.count()
    print(f"  Total records in vector database: {total_records}")
    
    # Check records with embeddings
    records_with_vectors = 0
    records_without_vectors = 0
    
    all_records = await vector_db.find()
    
    for record in all_records:
        if "embedding" in record and record["embedding"]:
            records_with_vectors += 1
        else:
            records_without_vectors += 1
    
    print(f"  Records with vectors: {records_with_vectors}")
    print(f"  Records without vectors: {records_without_vectors}")
    
    # Test vector search
    if records_with_vectors > 0:
        print("\n  Testing vector search capability...")
        
        query_text = "container orchestration and deployment"
        query_embedding = generate_embedding(query_text)
        
        results = await vector_db.vector_search(
            query_vector=query_embedding,
            k=3,
            vector_field="embedding"
        )
        
        if results:
            print(f"  ✓ Vector search working! Found {len(results)} results")
            for i, result in enumerate(results[:2], 1):
                print(f"    {i}. {result.record.get('title', 'Unknown')} (Score: {result.score:.3f})")
        else:
            print("  ⚠ No vector search results found")
    
    return {
        "total": total_records,
        "with_vectors": records_with_vectors,
        "without_vectors": records_without_vectors
    }


async def migration_with_retry(legacy_db, vector_db):
    """Demonstrate migration with retry logic for failed records."""
    
    print("\n6. Migration with Retry Logic...")
    
    # Create migration with simplified API
    migration = VectorMigration(
        source_db=legacy_db,
        target_db=vector_db,
        embedding_fn=generate_embedding,
        text_fields=["title", "content"],  # Primary configuration
        vector_field="embedding",
        batch_size=2
    )
    
    # Add retry logic
    max_retries = 3
    retry_delay = 0.5  # seconds
    
    # Run migration with simplified API
    print(f"  Running migration...")
    
    results = await migration.run(
        progress_callback=lambda status: print(f"    Progress: {status.migrated_records}/{status.total_records}")
    )
    
    print(f"  ✓ Migration completed")
    print(f"    Success: {results.migrated_records} records")
    print(f"    Failed: {results.failed_records} records")
    print(f"    Success rate: {results.success_rate:.1f}%")
    
    return results


async def main():
    """Run the migration example."""
    
    print("\n" + "="*60)
    print("Vector Migration Example")
    print("="*60)
    
    # Track overall statistics
    stats = MigrationStats()
    stats.start_time = time.time()
    
    try:
        # Step 1: Create legacy database
        legacy_db = await create_legacy_database()
        
        # Step 2: Create vector-enabled database
        vector_db = await migrate_to_vector_database(legacy_db)
        
        # Step 3: Manual migration
        failed_records = await manual_migration(legacy_db, vector_db, stats)
        
        if failed_records:
            print(f"\n⚠ Warning: {len(failed_records)} records failed to migrate")
            print("  Retrying failed records...")
            
            # Retry failed records
            for failed in failed_records:
                try:
                    record = failed["record"]
                    text = f"{record.get('title', '')} {record.get('content', '')}"
                    embedding = generate_embedding(text)
                    
                    new_record = Record({
                        **record,
                        "embedding": VectorField(embedding, dimensions=384),
                        "migrated_at": datetime.now().isoformat(),
                        "retry": True
                    })
                    
                    await vector_db.create(new_record)
                    stats.migrated_records += 1
                    stats.failed_records -= 1
                    print(f"    ✓ Retry successful: {record.get('title', 'Unknown')[:50]}")
                    
                except Exception as e:
                    print(f"    ✗ Retry failed: {e}")
        
        # Step 4: Incremental vectorization (for additional embeddings)
        await incremental_migration(vector_db)
        
        # Step 5: Verify migration
        verification = await verify_migration(vector_db)
        
        # Final statistics
        stats.end_time = time.time()
        
        print("\n" + "="*60)
        print("Migration Complete - Summary")
        print("="*60)
        print(f"  Total Records: {stats.total_records}")
        print(f"  Successfully Migrated: {stats.migrated_records}")
        print(f"  Failed: {stats.failed_records}")
        print(f"  Success Rate: {stats.success_rate:.1f}%")
        print(f"  Duration: {stats.duration:.2f} seconds")
        print(f"  Average Speed: {stats.migrated_records/stats.duration:.1f} records/second")
        
        # Cleanup
        await legacy_db.close()
        await vector_db.close()
        
        print("\n✓ Migration example completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())