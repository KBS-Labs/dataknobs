from dataknobs_data.factory import DatabaseFactory
from dataknobs_data.records import Record
from dataknobs_fsm.streaming.db_stream import DatabaseStreamSource

# Create test database
factory = DatabaseFactory()
db = factory.create(backend="memory")

# Add test records
for i in range(5):
    record = Record(id=str(i), data={'value': i})
    db.create(record)

# Create stream source
source = DatabaseStreamSource(
    database=db,
    batch_size=2
)

# Read chunks
chunks = list(source)

print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk.data)} items, is_last={chunk.is_last}")
    if chunk.metadata.get('error'):
        print(f"  Error: {chunk.metadata['error']}")
