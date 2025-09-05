from dataknobs_data.factory import DatabaseFactory
from dataknobs_data.records import Record
from dataknobs_data.query import Query

# Create test database
factory = DatabaseFactory()
db = factory.create(backend="memory")

# Add test records
for i in range(5):
    record = Record(id=str(i), data={'value': i})
    db.create(record)

# Test Query with limit
query = Query()
query.limit = 2
print(f"Query with limit: {query}")
print(f"Query limit value: {query.limit}")

# Search with query
results = db.search(query)
print(f"Results with limit=2: {len(results)} records")

# Try to get cursor value from last record
if results:
    last = results[-1]
    print(f"Last record: {last}")
    print(f"Last record id: {last.id}")
    print(f"Has 'field' attr: {hasattr(last, 'field')}")
    print(f"Dir of record: {[a for a in dir(last) if not a.startswith('_')]}")
