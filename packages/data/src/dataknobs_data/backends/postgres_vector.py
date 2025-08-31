"""PostgreSQL vector support utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg
    import numpy as np

logger = logging.getLogger(__name__)


def check_pgvector_extension_sync(db: Any) -> bool:
    """Check if pgvector extension is installed (sync version).
    
    Args:
        db: PostgresDB connection object
        
    Returns:
        True if pgvector is installed, False otherwise
    """
    try:
        result = db.query("""
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            ) as exists
        """)
        return bool(result.iloc[0]["exists"]) if not result.empty else False
    except Exception as e:
        logger.debug(f"Could not check pgvector extension: {e}")
        return False


def install_pgvector_extension_sync(db: Any) -> bool:
    """Install pgvector extension if not already installed (sync version).
    
    Args:
        db: PostgresDB connection object
        
    Returns:
        True if installation successful or already installed
    """
    try:
        # Check if already installed
        if check_pgvector_extension_sync(db):
            logger.debug("pgvector extension already installed")
            return True

        # Try to install
        db.execute("CREATE EXTENSION IF NOT EXISTS vector")
        logger.info("Successfully installed pgvector extension")
        return True
    except Exception as e:
        logger.warning(f"Could not install pgvector extension: {e}")
        return False


async def check_pgvector_extension(conn: asyncpg.Connection) -> bool:
    """Check if pgvector extension is installed.
    
    Args:
        conn: AsyncPG connection
        
    Returns:
        True if pgvector is installed, False otherwise
    """
    result = await conn.fetchval("""
        SELECT EXISTS (
            SELECT 1 FROM pg_extension WHERE extname = 'vector'
        )
    """)
    return bool(result)


async def install_pgvector_extension(conn: asyncpg.Connection) -> bool:
    """Install pgvector extension if not already installed.
    
    Args:
        conn: AsyncPG connection
        
    Returns:
        True if installation successful or already installed
    """
    try:
        # Check if already installed
        if await check_pgvector_extension(conn):
            logger.debug("pgvector extension already installed")
            return True

        # Try to install
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        logger.info("Successfully installed pgvector extension")
        return True
    except Exception as e:
        logger.warning(f"Could not install pgvector extension: {e}")
        return False


def get_vector_operator(metric: str) -> str:
    """Get PostgreSQL vector operator for distance metric.
    
    Args:
        metric: Distance metric (cosine, euclidean, inner_product)
        
    Returns:
        PostgreSQL operator string
    """
    operators = {
        "cosine": "<=>",  # Cosine distance
        "euclidean": "<->",  # L2 distance
        "inner_product": "<#>",  # Negative inner product
        "l2": "<->",  # Alias for euclidean
        "ip": "<#>",  # Alias for inner product
    }
    return operators.get(metric.lower(), "<=>")  # Default to cosine


def get_optimal_index_type(num_vectors: int) -> tuple[str, dict[str, Any]]:
    """Determine optimal index type based on dataset size.
    
    Args:
        num_vectors: Number of vectors in dataset
        
    Returns:
        Tuple of (index_type, index_parameters)
    """
    if num_vectors < 10000:
        # For small datasets, use IVFFlat with fewer lists
        return "ivfflat", {"lists": min(100, num_vectors // 10)}
    elif num_vectors < 1000000:
        # For medium datasets, use IVFFlat with standard parameters
        lists = int(num_vectors ** 0.5)  # Square root heuristic
        return "ivfflat", {"lists": min(lists, 5000)}
    else:
        # For large datasets, consider HNSW (if available in pgvector version)
        # Note: HNSW requires pgvector 0.5.0+
        return "hnsw", {"m": 16, "ef_construction": 200}


def build_vector_index_sql(
    table_name: str,
    schema_name: str,
    column_name: str,
    dimensions: int,
    metric: str = "cosine",
    index_type: str = "ivfflat",
    index_params: dict[str, Any] | None = None,
    field_name: str | None = None
) -> str:
    """Build SQL for creating a vector index.
    
    Args:
        table_name: Name of table
        schema_name: Schema name
        column_name: SQL expression for vector column
        dimensions: Vector dimensions
        metric: Distance metric
        index_type: Type of index (ivfflat, hnsw)
        index_params: Index-specific parameters
        field_name: Original field name for index naming
        
    Returns:
        SQL CREATE INDEX statement
    """
    index_params = index_params or {}

    # Determine field name for index naming
    if not field_name:
        field_name = extract_field_name(column_name)

    index_name = get_vector_index_name(table_name, field_name, metric)

    # Determine operator class based on metric
    op_class = {
        "cosine": "vector_cosine_ops",
        "euclidean": "vector_l2_ops",
        "l2": "vector_l2_ops",
        "inner_product": "vector_ip_ops",
        "ip": "vector_ip_ops",
        "dot_product": "vector_ip_ops",
    }.get(metric.lower(), "vector_cosine_ops")

    if index_type == "ivfflat":
        lists = index_params.get("lists", 100)
        # IVFFlat requires proper parentheses for functional indexes with operator class
        # The column_name should already include the dimension cast
        return f"""
        CREATE INDEX IF NOT EXISTS {index_name}
        ON {schema_name}.{table_name}
        USING ivfflat (({column_name}) {op_class})
        WITH (lists = {lists})
        """
    elif index_type == "hnsw":
        m = index_params.get("m", 16)
        ef_construction = index_params.get("ef_construction", 200)
        # HNSW index (requires pgvector 0.5.0+)
        # The column_name should already include the dimension cast
        return f"""
        CREATE INDEX IF NOT EXISTS {index_name}
        ON {schema_name}.{table_name}  
        USING hnsw (({column_name}) {op_class})
        WITH (m = {m}, ef_construction = {ef_construction})
        """
    else:
        # Default to basic index
        return f"""
        CREATE INDEX IF NOT EXISTS {index_name}
        ON {schema_name}.{table_name}
        USING btree ({column_name})
        """


def sanitize_identifier(name: str) -> str:
    """Sanitize a string to be used as a database identifier.
    
    Removes or replaces special characters that are not valid in identifiers.
    
    Args:
        name: Raw string that may contain special characters
        
    Returns:
        Sanitized string safe for use as identifier
    """
    import re
    # Remove SQL operators and special chars
    name = re.sub(r"[->()'\[\]:,\s]+", "_", name)
    # Remove multiple underscores
    name = re.sub(r"_+", "_", name)
    # Remove leading/trailing underscores
    name = name.strip("_")
    return name


def extract_field_name(column_expression: str) -> str:
    """Extract field name from a column expression.
    
    Args:
        column_expression: SQL expression like "(data->'field'->>'value')::vector"
        
    Returns:
        Extracted field name or 'vector' as fallback
    """
    import re
    # Try to extract from JSON path expressions
    patterns = [
        r"data->'([^']+)'",  # data->'field'
        r"data->>'([^']+)'",  # data->>'field'
        r"\$\.([^'\"]+)",  # $.field (JSONPath)
        r"'([^']+)'",  # Any quoted string
    ]

    for pattern in patterns:
        match = re.search(pattern, column_expression)
        if match:
            return match.group(1)

    # Fallback: try to use the whole expression after basic cleanup
    cleaned = sanitize_identifier(column_expression)
    return cleaned if cleaned else "vector"


def get_vector_index_name(table_name: str, field_name: str, metric: str = "cosine") -> str:
    """Generate consistent index name for vector field.
    
    Args:
        table_name: Name of the table
        field_name: Name of the vector field (or column expression)
        metric: Distance metric
        
    Returns:
        Index name string
    """
    # Sanitize all parts
    clean_table = sanitize_identifier(table_name)
    clean_field = sanitize_identifier(field_name)
    clean_metric = sanitize_identifier(metric)

    return f"idx_{clean_table}_{clean_field}_{clean_metric}"


def build_vector_column_expression(field_name: str, dimensions: int | None = None, for_index: bool = False) -> str:
    """Build SQL expression for vector column from JSON field.
    
    Args:
        field_name: Name of the vector field in JSON
        dimensions: Optional dimensions for casting
        for_index: Whether this is for index creation (needs special handling)
        
    Returns:
        SQL expression for vector column
    """
    dim_cast = f"({dimensions})" if dimensions else ""

    if for_index:
        # For indexes, we need a simpler expression
        # Since we're storing VectorFields as objects with 'value' key, index on that
        return f"(data->'{field_name}'->>'value')::vector{dim_cast}"
    else:
        # For queries, we can use the same expression
        return f"(data->'{field_name}'->>'value')::vector{dim_cast}"


def get_vector_count_sql(schema_name: str, table_name: str, field_name: str) -> str:
    """Get SQL to count vectors in a field.
    
    Args:
        schema_name: Database schema
        table_name: Table name
        field_name: Vector field name
        
    Returns:
        SQL query string
    """
    return f"""
    SELECT COUNT(*) as count 
    FROM {schema_name}.{table_name}
    WHERE data ? '{field_name}'
    """


def get_index_check_sql(schema_name: str, table_name: str, field_name: str) -> tuple[str, list[Any]]:
    """Get SQL to check if vector index exists.
    
    Args:
        schema_name: Database schema
        table_name: Table name  
        field_name: Vector field name
        
    Returns:
        Tuple of (SQL query, parameters)
    """
    sql = """
    SELECT COUNT(*) > 0 as has_index
    FROM pg_indexes
    WHERE schemaname = $1
    AND tablename = $2
    AND indexname LIKE $3
    """
    index_pattern = f"%{field_name}%"
    return sql, [schema_name, table_name, index_pattern]


def format_vector_for_postgres(vector: np.ndarray | list[float]) -> str:
    """Format vector for PostgreSQL vector column.
    
    Args:
        vector: Numpy array or list of floats
        
    Returns:
        PostgreSQL vector string format
    """
    if hasattr(vector, 'tolist'):
        vector = vector.tolist()

    # Format as PostgreSQL vector literal
    return f"[{','.join(str(float(v)) for v in vector)}]"


def parse_postgres_vector(vector_str: str) -> list[float]:
    """Parse PostgreSQL vector string to list of floats.
    
    Args:
        vector_str: PostgreSQL vector string like '[0.1,0.2,0.3]'
        
    Returns:
        List of floats
    """
    if not vector_str or vector_str == "[]":
        return []

    # Remove brackets and split by comma
    vector_str = vector_str.strip("[]")
    return [float(v.strip()) for v in vector_str.split(",")]
