#!/usr/bin/env python3
"""
Vector Support for Multiple Backends Example

This example demonstrates vector search capabilities across different backends:
1. Memory backend - for fast in-memory vector operations
2. File backend - for persistent local storage with vectors
3. S3 backend - for cloud storage with vector support

Each backend uses Python-based vector similarity calculations,
making vector search available regardless of the storage mechanism.

Note: While functional, the S3 and File backends download all records
for vector search, which may be inefficient for large datasets.
Consider PostgreSQL or Elasticsearch for production use with large vector datasets.
"""

import asyncio
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from dataknobs_data import DatabaseFactory, AsyncDatabaseFactory
from dataknobs_data.records import Record
from dataknobs_data.fields import Field, VectorField
from dataknobs_data.query import Query, Operator


class MultiBackendVectorExample:
    """Demonstrates vector search across multiple backend types."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the example."""
        self.verbose = verbose
        self.test_data = self._create_test_vectors()
    
    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def _create_test_vectors(self) -> List[Dict[str, Any]]:
        """Create test data with 3D vectors for geometric understanding."""
        return [
            {
                "name": "X-axis",
                "description": "Unit vector pointing along X axis",
                "vector": np.array([1.0, 0.0, 0.0]),
                "category": "axis"
            },
            {
                "name": "Y-axis", 
                "description": "Unit vector pointing along Y axis",
                "vector": np.array([0.0, 1.0, 0.0]),
                "category": "axis"
            },
            {
                "name": "Z-axis",
                "description": "Unit vector pointing along Z axis", 
                "vector": np.array([0.0, 0.0, 1.0]),
                "category": "axis"
            },
            {
                "name": "XY-diagonal",
                "description": "Vector in XY plane",
                "vector": np.array([0.707, 0.707, 0.0]),
                "category": "diagonal"
            },
            {
                "name": "XZ-diagonal",
                "description": "Vector in XZ plane",
                "vector": np.array([0.707, 0.0, 0.707]),
                "category": "diagonal"
            },
            {
                "name": "YZ-diagonal",
                "description": "Vector in YZ plane",
                "vector": np.array([0.0, 0.707, 0.707]),
                "category": "diagonal"
            }
        ]
    
    def create_records(self) -> List[Record]:
        """Create Record objects from test data."""
        records = []
        for i, item in enumerate(self.test_data):
            # Normalize vector
            vec = item["vector"]
            vec = vec / np.linalg.norm(vec)
            
            record = Record(
                data={
                    "name": Field(name="name", value=item["name"]),
                    "description": Field(name="description", value=item["description"]),
                    "category": Field(name="category", value=item["category"]),
                    "embedding": VectorField(
                        name="embedding",
                        value=vec,
                        dimensions=3
                    )
                },
                metadata={"index": i}
            )
            records.append(record)
        return records
    
    def run_sync_memory_example(self) -> List[Any]:
        """Demonstrate vector search with sync Memory backend."""
        self.log("\n=== Memory Backend (Sync) ===")
        self.log("Fast in-memory vector operations")
        
        # Create database with vector support
        factory = DatabaseFactory()
        db = factory.create(
            backend="memory",
            vector_enabled=True,
            vector_metric="cosine"
        )
        
        # Connect and populate
        db.connect()
        records = self.create_records()
        ids = []
        for record in records:
            id = db.create(record)
            ids.append(id)
            self.log(f"  Created: {record.get_field('name').value}")
        
        # Perform vector search - find vectors similar to XY diagonal
        query_vector = np.array([0.7, 0.7, 0.0])
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        results = db.vector_search(
            query_vector=query_vector,
            vector_field="embedding",
            k=3
        )
        
        self.log(f"\nSearching for vectors similar to [0.7, 0.7, 0.0]:")
        for result in results:
            name = result.record.get_field('name').value
            self.log(f"  - {name}: similarity={result.score:.3f}")
        
        # Clean up
        db.clear()
        return results
    
    def run_sync_file_example(self, filepath: Optional[str] = None) -> List[Any]:
        """Demonstrate vector search with sync File backend."""
        self.log("\n=== File Backend (Sync) ===")
        self.log("Persistent local storage with vector support")
        
        # Use provided path or create temporary file
        temp_file = None
        cleanup_file = False
        if not filepath:
            temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            filepath = temp_file.name
            temp_file.close()
            cleanup_file = True
        
        try:
            # Create database with vector support
            factory = DatabaseFactory()
            db = factory.create(
                backend="file",
                path=filepath,
                vector_enabled=True,
                vector_metric="euclidean"
            )
            
            # Connect and populate
            db.connect()
            records = self.create_records()
            for record in records:
                db.create(record)
            self.log(f"  Stored {len(records)} records to {filepath}")
            
            # Perform vector search with different metric
            query_vector = np.array([0.0, 0.0, 1.0])  # Z-axis
            
            results = db.vector_search(
                query_vector=query_vector,
                vector_field="embedding",
                k=3,
                metric="euclidean"  # Using Euclidean distance
            )
            
            self.log(f"\nSearching for vectors closest to Z-axis [0, 0, 1]:")
            for result in results:
                name = result.record.get_field('name').value
                self.log(f"  - {name}: distance_score={result.score:.3f}")
            
            # Clean up database content
            db.clear()
            return results
            
        finally:
            # Clean up file if temporary
            if cleanup_file and os.path.exists(filepath):
                os.unlink(filepath)
    
    async def run_async_memory_example(self) -> List[Any]:
        """Demonstrate vector search with async Memory backend."""
        self.log("\n=== Memory Backend (Async) ===")
        self.log("Async in-memory vector operations")
        
        # Create database with vector support
        factory = AsyncDatabaseFactory()
        db = factory.create(
            backend="memory",
            vector_enabled=True,
            vector_metric="dot_product"
        )
        
        # Connect and populate
        await db.connect()
        records = self.create_records()
        for record in records:
            await db.create(record)
        self.log(f"  Created {len(records)} vector records")
        
        # Perform vector search with dot product metric
        query_vector = np.array([1.0, 0.0, 0.0])  # X-axis
        
        results = await db.vector_search(
            query_vector=query_vector,
            vector_field="embedding",
            k=3,
            metric="dot_product"
        )
        
        self.log(f"\nSearching using dot product similarity to X-axis [1, 0, 0]:")
        for result in results:
            name = result.record.get_field('name').value
            self.log(f"  - {name}: dot_product={result.score:.3f}")
        
        # Clean up
        await db.clear()
        return results
    
    async def run_async_file_with_filter(self, filepath: Optional[str] = None) -> List[Any]:
        """Demonstrate async vector search with filtering."""
        self.log("\n=== File Backend (Async) with Filtering ===")
        self.log("Combining vector search with metadata filters")
        
        # Use provided path or create temporary file
        temp_file = None
        cleanup_file = False
        if not filepath:
            temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            filepath = temp_file.name
            temp_file.close()
            cleanup_file = True
        
        try:
            # Create database
            factory = AsyncDatabaseFactory()
            db = factory.create(
                backend="file",
                path=filepath,
                vector_enabled=True,
                vector_metric="cosine"
            )
            
            # Connect and populate
            await db.connect()
            records = self.create_records()
            for record in records:
                await db.create(record)
            
            # Create filter query - only consider "diagonal" category
            filter_query = Query().filter("category", Operator.EQ, "diagonal")
            
            # Search for vectors similar to XY diagonal, but only among diagonals
            query_vector = np.array([1.0, 1.0, 0.0])
            query_vector = query_vector / np.linalg.norm(query_vector)
            
            results = await db.vector_search(
                query_vector=query_vector,
                vector_field="embedding",
                k=2,
                filter=filter_query
            )
            
            self.log(f"\nFiltered search (category='diagonal') for [0.7, 0.7, 0]:")
            for result in results:
                name = result.record.get_field('name').value
                category = result.record.get_field('category').value
                self.log(f"  - {name} ({category}): score={result.score:.3f}")
            
            # Clean up
            await db.clear()
            return results
            
        finally:
            # Clean up file if temporary
            if cleanup_file and os.path.exists(filepath):
                os.unlink(filepath)
    
    def run_s3_example_info(self):
        """Provide information about S3 backend vector support."""
        self.log("\n=== S3 Backend Vector Support ===")
        self.log("S3 backend also supports vector search with the same Python-based approach.")
        self.log("\nTo use S3 backend with vectors:")
        self.log("1. Set up AWS credentials or use LocalStack for testing")
        self.log("2. Create database with backend='s3' and vector_enabled=True")
        self.log("3. Use vector_search() method just like other backends")
        self.log("\nExample configuration:")
        self.log("  db = factory.create(")
        self.log("    backend='s3',")
        self.log("    bucket='my-bucket',")
        self.log("    prefix='vectors/',")
        self.log("    vector_enabled=True,")
        self.log("    vector_metric='cosine'")
        self.log("  )")
        self.log("\nNote: S3 backend downloads all records for vector search,")
        self.log("      which may be slow for large datasets.")


def main():
    """Run all examples."""
    example = MultiBackendVectorExample(verbose=True)
    
    print("=" * 60)
    print("Vector Support for Multiple Backends")
    print("=" * 60)
    
    # Run sync examples
    example.run_sync_memory_example()
    example.run_sync_file_example()
    
    # Run async examples
    asyncio.run(example.run_async_memory_example())
    asyncio.run(example.run_async_file_with_filter())
    
    # Show S3 information
    example.run_s3_example_info()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()