"""Examples of using the simplified schema system."""

import asyncio
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.fields import FieldType
from dataknobs_data.records import Record
from dataknobs_data.schema import DatabaseSchema


async def example_1_fluent_api():
    """Example using fluent API for schema definition."""
    # Create database with inline schema using fluent API
    db = AsyncMemoryDatabase().with_schema(
        content=FieldType.TEXT,
        embedding=(FieldType.VECTOR, {"dimensions": 384, "source_field": "content"}),
        title=FieldType.TEXT,
        title_embedding=(FieldType.VECTOR, {"dimensions": 384, "source_field": "title"}),
        score=FieldType.FLOAT,
    )
    await db.connect()
    
    # The schema is now available
    print("Vector fields:", db.schema.get_vector_fields().keys())
    print("Source field mappings:", db.schema.get_source_fields())
    
    await db.close()


async def example_2_builder_pattern():
    """Example using builder pattern for schema."""
    # Build schema using chainable methods
    schema = (DatabaseSchema()
        .add_text_field("content")
        .add_vector_field("embedding", dimensions=384, source_field="content")
        .add_text_field("title")
        .add_vector_field("title_embedding", dimensions=384, source_field="title")
    )
    
    db = AsyncMemoryDatabase()
    db.set_schema(schema)
    await db.connect()
    
    print("Schema fields:", list(schema.fields.keys()))
    
    await db.close()


async def example_3_config_based():
    """Example using config-based initialization."""
    config = {
        "schema": {
            "fields": {
                "content": "text",
                "embedding": {
                    "type": "vector",
                    "dimensions": 384,
                    "source_field": "content"
                },
                "title": "text",
                "score": {"type": "float", "required": True}
            }
        },
        # Other config options can go here
        "cache_size": 1000,
    }
    
    db = AsyncMemoryDatabase(config)
    await db.connect()
    
    print("Database schema loaded from config")
    print("Vector fields:", db.schema.get_vector_fields().keys())
    
    # The cache_size config is still available to the backend
    print("Cache size config:", db.config.get("cache_size"))
    
    await db.close()


async def example_4_with_synchronizer():
    """Example using schema with vector synchronization."""
    from dataknobs_data.vector.sync import VectorTextSynchronizer
    import numpy as np
    
    # Simple embedding function for demo
    def embed_text(text: str) -> np.ndarray:
        return np.random.rand(384)
    
    # Create database with schema
    db = AsyncMemoryDatabase().with_schema(
        content=FieldType.TEXT,
        embedding=(FieldType.VECTOR, {"dimensions": 384, "source_field": "content"}),
    )
    await db.connect()
    
    # Create synchronizer - it will use the schema automatically
    sync = VectorTextSynchronizer(
        database=db,
        embedding_fn=embed_text,
        model_name="demo-model",
        model_version="v1.0"
    )
    
    # Add a record
    record = Record(data={"content": "Hello, world!"})
    record_id = await db.create(record)
    
    # Sync will automatically detect vector fields from schema
    record = await db.read(record_id)
    success, updated_fields = await sync.sync_record(record)
    
    if success:
        await db.update(record_id, record)
        print(f"Updated fields: {updated_fields}")
        print(f"Embedding shape: {len(record.data['embedding'])}")
    
    await db.close()


if __name__ == "__main__":
    asyncio.run(example_1_fluent_api())
    asyncio.run(example_2_builder_pattern())
    asyncio.run(example_3_config_based())
    asyncio.run(example_4_with_synchronizer())