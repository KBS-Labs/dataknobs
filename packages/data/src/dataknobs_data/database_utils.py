"""Utility functions for database operations."""


from .query import Query
from .records import Record


def ensure_record_id(record: Record, record_id: str) -> Record:
    """Ensure a record has its storage ID set.
    
    Helper method for backends to ensure records have their storage IDs set
    when returning from storage operations like search.
    
    Args:
        record: The record to check
        record_id: The storage ID that should be set on the record
        
    Returns:
        The record with storage_id guaranteed to be set
    """
    if not record.has_storage_id() or record.storage_id != record_id:
        record = record.copy(deep=True)
        record.storage_id = record_id
    return record


def process_search_results(
    results: list[tuple[str, Record]],
    query: Query,
    deep_copy: bool = True
) -> list[Record]:
    """Process search results with standard operations.
    
    Helper method for backends to process search results consistently:
    1. Ensures records have their IDs set
    2. Applies sorting
    3. Applies offset and limit
    4. Applies field projection
    5. Returns deep copies if requested
    
    Args:
        results: List of (id, record) tuples
        query: The query with sorting, pagination, and projection specs
        deep_copy: Whether to return deep copies of records
        
    Returns:
        Processed list of records
    """
    # Apply sorting
    if query.sort_specs:
        for sort_spec in reversed(query.sort_specs):
            reverse = sort_spec.order.value == "desc"
            # Special handling for 'id' field
            if sort_spec.field == 'id':
                results.sort(
                    key=lambda x: x[0] or "",  # x[0] is the record ID
                    reverse=reverse
                )
            else:
                results.sort(
                    key=lambda x: x[1].get_value(sort_spec.field, ""),
                    reverse=reverse
                )

    # Extract records and ensure they have their IDs
    records = []
    for record_id, record in results:
        processed_record = ensure_record_id(record, record_id)
        if deep_copy:
            processed_record = processed_record.copy(deep=True)
        records.append(processed_record)

    # Apply offset and limit
    if query.offset_value:
        records = records[query.offset_value:]
    if query.limit_value:
        records = records[:query.limit_value]

    # Apply field projection
    if query.fields:
        records = [record.project(query.fields) for record in records]

    return records
