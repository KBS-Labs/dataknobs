"""PostgreSQL vector support utilities."""

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg
    import numpy as np

logger = logging.getLogger(__name__)


async def check_pgvector_extension(conn: "asyncpg.Connection") -> bool:
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


async def install_pgvector_extension(conn: "asyncpg.Connection") -> bool:
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
    index_params: dict[str, Any] | None = None
) -> str:
    """Build SQL for creating a vector index.
    
    Args:
        table_name: Name of table
        schema_name: Schema name
        column_name: Name of vector column
        dimensions: Vector dimensions
        metric: Distance metric
        index_type: Type of index (ivfflat, hnsw)
        index_params: Index-specific parameters
        
    Returns:
        SQL CREATE INDEX statement
    """
    index_params = index_params or {}
    operator = get_vector_operator(metric)
    index_name = f"idx_{table_name}_{column_name}_{metric}"
    
    if index_type == "ivfflat":
        lists = index_params.get("lists", 100)
        # IVFFlat requires specifying lists parameter
        return f"""
        CREATE INDEX IF NOT EXISTS {index_name}
        ON {schema_name}.{table_name}
        USING ivfflat ({column_name} {operator})
        WITH (lists = {lists})
        """
    elif index_type == "hnsw":
        m = index_params.get("m", 16)
        ef_construction = index_params.get("ef_construction", 200)
        # HNSW index (requires pgvector 0.5.0+)
        return f"""
        CREATE INDEX IF NOT EXISTS {index_name}
        ON {schema_name}.{table_name}  
        USING hnsw ({column_name} {operator})
        WITH (m = {m}, ef_construction = {ef_construction})
        """
    else:
        # Default to basic index
        return f"""
        CREATE INDEX IF NOT EXISTS {index_name}
        ON {schema_name}.{table_name}
        USING btree ({column_name})
        """


def format_vector_for_postgres(vector: "np.ndarray | list[float]") -> str:
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