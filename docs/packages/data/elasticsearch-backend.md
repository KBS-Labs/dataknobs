# Elasticsearch Backend

## Overview

The Elasticsearch Backend provides powerful full-text search, analytics, and distributed storage capabilities for large-scale applications.

## Features

- **Full-text search** - Advanced text analysis
- **Distributed** - Horizontal scaling
- **Real-time** - Near real-time indexing
- **Analytics** - Aggregations and facets
- **Both sync and async** - Using elasticsearch-py

## Configuration

```python
from dataknobs_data import ElasticsearchDatabase

config = {
    "hosts": ["http://localhost:9200"],
    "index": "records",
    "auth": ("elastic", "password"),
    "verify_certs": True,
    "pool_size": 10
}

db = ElasticsearchDatabase(config)
```

## Index Mapping

```json
{
  "mappings": {
    "properties": {
      "id": {"type": "keyword"},
      "fields": {"type": "object", "dynamic": true},
      "metadata": {"type": "object", "dynamic": true},
      "created_at": {"type": "date"},
      "updated_at": {"type": "date"}
    }
  }
}
```

## Usage Examples

### Full-Text Search

```python
# Text search across all fields
query = Query(filters=[
    Filter("_all", Operator.CONTAINS, "sensor malfunction")
])

results = db.search(query)
```

### Aggregations

```python
# Get statistics
aggregation = {
    "avg_temperature": {"avg": {"field": "fields.temperature"}},
    "max_humidity": {"max": {"field": "fields.humidity"}},
    "sensor_count": {"cardinality": {"field": "fields.sensor_id"}}
}

stats = db.aggregate(aggregation)
```

### Bulk Operations

```python
# Efficient bulk indexing
records = [Record(data) for data in dataset]

# Bulk index with refresh
db.create_batch(records, refresh="wait_for")
```

## Search Features

### Fuzzy Matching

```python
# Fuzzy search for typos
query = Query(filters=[
    Filter("name", Operator.FUZZY, "jhon")  # Matches "john"
])
```

### Geo Queries

```python
# Geo-distance queries
query = Query(filters=[
    Filter("location", Operator.GEO_DISTANCE, {
        "point": {"lat": 40.7128, "lon": -74.0060},
        "distance": "10km"
    })
])
```

## Performance Tuning

- **Sharding** - Distribute data across nodes
- **Replicas** - For high availability
- **Refresh interval** - Balance speed vs resources
- **Bulk size** - Optimize batch operations
- **Query cache** - Enable for repeated queries

## Cluster Management

```python
# Check cluster health
health = db.cluster_health()
print(f"Status: {health['status']}")
print(f"Nodes: {health['number_of_nodes']}")

# Index statistics
stats = db.index_stats()
print(f"Document count: {stats['doc_count']}")
print(f"Index size: {stats['size_in_bytes']}")
```

## Production Considerations

- **Monitoring** - Use Kibana or Grafana
- **Snapshots** - Regular backup snapshots
- **Security** - Enable X-Pack security
- **Scaling** - Plan shard allocation
- **Version** - Keep Elasticsearch updated

## See Also

- [Backends Overview](backends.md)
- [Query System](query.md)
- [Async Pooling](async-pooling.md)