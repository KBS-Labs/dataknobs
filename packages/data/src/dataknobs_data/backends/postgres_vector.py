# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""PostgreSQL vector support utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dataknobs_utils.sql_utils import quote_ident

from ..vector.types import DistanceMetric
from .sql_base import SQLRecordSerializer

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


# Keyed on ``DistanceMetric.canonical()``, so there are four keys and all six
# members reach one. The table this replaces was keyed on the spelling and had
# five entries, two of which were aliases; ``DOT_PRODUCT`` and ``L1`` fell
# through its ``.get(metric, "<=>")`` default and were answered in cosine
# distances. A missing key here is a ``KeyError`` at the lookup, not a wrong
# answer downstream --- and the exhaustiveness test in
# ``test_distance_metric_vocabulary.py`` reaches it before a caller does.
_OPERATORS: dict[DistanceMetric, str] = {
    DistanceMetric.COSINE: "<=>",  # cosine distance
    DistanceMetric.EUCLIDEAN: "<->",  # L2 distance
    DistanceMetric.DOT_PRODUCT: "<#>",  # negative inner product
    DistanceMetric.L1: "<+>",  # taxicab distance; pgvector 0.7.0+
}

_OPCLASSES: dict[DistanceMetric, str] = {
    DistanceMetric.COSINE: "vector_cosine_ops",
    DistanceMetric.EUCLIDEAN: "vector_l2_ops",
    DistanceMetric.DOT_PRODUCT: "vector_ip_ops",
    DistanceMetric.L1: "vector_l1_ops",
}


def get_vector_operator(metric: DistanceMetric | str) -> str:
    """The pgvector distance operator for a metric.

    Args:
        metric: A member, member value, or published alias --- whatever
            :meth:`DistanceMetric.resolve` accepts.

    Returns:
        The pgvector operator, one of ``<=>``, ``<->``, ``<#>``, ``<+>``.

    Raises:
        ValueError: If the name is not an accepted spelling. The table this
            replaces answered an unrecognised name --- and two recognised
            members --- with the cosine operator, so a search asking for dot
            product was silently ranked by cosine distance.
    """
    return _OPERATORS[DistanceMetric.resolve(metric).canonical()]


def get_vector_opclass(metric: DistanceMetric | str) -> str:
    """The pgvector index operator class for a metric.

    Separate from :func:`get_vector_operator` because they are different
    pgvector vocabularies, but derived from the same canonical member so the
    two cannot disagree about a spelling again --- which they did, the
    operator table knowing ``inner_product`` and the operator-class table
    knowing ``dot_product`` as well.

    Args:
        metric: A member, member value, or published alias.

    Returns:
        The operator class name, e.g. ``vector_cosine_ops``.

    Raises:
        ValueError: If the name is not an accepted spelling.
    """
    return _OPCLASSES[DistanceMetric.resolve(metric).canonical()]


def distance_to_score(metric: DistanceMetric | str, distance: float) -> float:
    """Convert a pgvector distance into a similarity score.

    One conversion, because there were two and they disagreed. Each Postgres
    twin carried its own copy: for cosine the sync twin returned ``1 - d``
    (the cosine similarity, which is also what the ten Python-path backends
    return) and the async twin returned ``1 - min(d, 2) / 2``, a different
    number for the same corpus and the same query. Nothing compared the two
    until ``score_threshold`` arrived to compare either against a constant.

    Args:
        metric: The metric the distance was measured under.
        distance: What the pgvector operator returned.

    Returns:
        A score that rises as the neighbour gets nearer, comparable across
        backends for the same metric.

    Raises:
        ValueError: If the name is not an accepted spelling.
    """
    canonical = DistanceMetric.resolve(metric).canonical()
    if canonical is DistanceMetric.COSINE:
        # pgvector's cosine distance is ``1 - cosine_similarity`` over [0, 2],
        # so this is the similarity itself rather than a rescaling of it.
        return 1.0 - distance
    if canonical is DistanceMetric.DOT_PRODUCT:
        # ``<#>`` is the *negative* inner product, so negating recovers it.
        return -distance
    # Euclidean and L1 are unbounded above; map to (0, 1] preserving order.
    return 1.0 / (1.0 + distance)


def build_vector_value_expression(field_name: str, dimensions: int | None = None) -> str:
    """The one SQL expression for the vector stored in ``field_name``.

    Used by both twins' searches and by index creation, because an expression
    index serves only a query whose expression matches exactly and these were
    three different expressions. The search spelled it
    ``SQLRecordSerializer.get_vector_extraction_sql`` --- a ``CASE`` tolerating
    both the ``VectorField`` object form and a bare array --- while index
    creation spelled it ``(data->'f'->>'value')::vector(n)``, only the object
    form and with a width. No query could use that index, and the twin that
    would have wanted it could not create one.

    Args:
        field_name: The record field holding the vector. Validated against the
            JSONB-key grammar by the serializer, since it lands in a SQL
            string literal where ``quote_ident`` does not apply.
        dimensions: When given, the expression is cast to ``vector(n)``.
            pgvector will not index an expression of unfixed width, and the
            query must carry the same cast or the planner sees two different
            expressions.

    Returns:
        A parenthesised SQL expression yielding a ``vector``.
    """
    inner = SQLRecordSerializer.get_vector_extraction_sql(field_name, dialect="postgres")
    if dimensions is None:
        return f"({inner})"
    return f"(({inner})::vector({dimensions}))"


def build_vector_search_sql(
    *,
    q_qualified: str,
    vector_field: str,
    dimensions: int,
    metric: DistanceMetric | str,
    vector_placeholder: str,
    field_placeholder: str,
    limit_clause: str,
    filter_clause: str = "",
) -> str:
    """Build the k-NN query both Postgres twins run.

    Written once because the two twins hand-built nearly this same statement
    and, in doing so, pointed them at different storage: one at the JSON
    ``data`` column every write path fills, the other at a ``vector_<field>``
    column that three of fourteen write paths filled. They differ now only in
    how their driver spells a placeholder, which is what the three placeholder
    arguments carry.

    Args:
        q_qualified: The pre-quoted ``"schema"."table"``.
        vector_field: The record field holding the vector.
        dimensions: Width of the query vector, which is necessarily the width
            of the stored ones --- pgvector refuses to compare across widths.
            Passing it lets the expression match an index built for that width.
        metric: The metric to rank under.
        vector_placeholder: The driver's placeholder for the query vector,
            e.g. ``%(p0)s`` or ``$1``.
        field_placeholder: The driver's placeholder for the field name, used
            by the ``data ? <field>`` existence test.
        limit_clause: A complete ``LIMIT ...`` clause, since asyncpg and
            psycopg2 differ on whether ``k`` may be bound.
        filter_clause: An optional ``AND ...`` fragment from the query builder.

    Returns:
        The SELECT statement, yielding ``id``, ``data``, ``metadata`` and
        ``distance``.
    """
    expression = build_vector_value_expression(vector_field, dimensions)
    operator = get_vector_operator(metric)
    return f"""
        SELECT
            id,
            data,
            metadata,
            {expression} {operator} {vector_placeholder}::vector({dimensions}) AS distance
        FROM {q_qualified}
        WHERE data ? {field_placeholder}
        {filter_clause}
        ORDER BY distance
        {limit_clause}
    """


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
        lists = int(num_vectors**0.5)  # Square root heuristic
        return "ivfflat", {"lists": min(lists, 5000)}
    else:
        # For large datasets, consider HNSW (if available in pgvector version)
        # Note: HNSW requires pgvector 0.5.0+
        return "hnsw", {"m": 16, "ef_construction": 200}


def build_vector_index_sql(
    q_table_name: str,
    q_schema_name: str,
    column_name: str,
    dimensions: int,
    metric: DistanceMetric | str = DistanceMetric.COSINE,
    index_type: str = "ivfflat",
    index_params: dict[str, Any] | None = None,
    field_name: str | None = None,
) -> str:
    """Build SQL for creating a vector index.

    Args:
        q_table_name: Pre-quoted table name (e.g. ``'"MyTable"'``)
        q_schema_name: Pre-quoted schema name (e.g. ``'"public"'``)
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

    # Derive the raw table name for index naming.  q_table_name is assumed to be
    # a quote_ident-produced value (starts and ends with '"'); anything else is
    # passed through unchanged so callers with plain names still work.
    raw_table_name = (
        q_table_name[1:-1].replace('""', '"')
        if q_table_name.startswith('"') and q_table_name.endswith('"')
        else q_table_name
    )
    # Canonicalised, so the two spellings of one metric name one index.
    # ``drop_vector_index`` rebuilds the name from its own ``metric`` argument
    # and would otherwise miss an index created under the other spelling.
    metric_name = DistanceMetric.resolve(metric).canonical().value
    index_name = get_vector_index_name(raw_table_name, field_name, metric_name)
    # Quote the index name so it is consistent with drop_vector_index, which
    # already calls quote_ident(index_name).  Without quoting, PostgreSQL folds
    # the name to lowercase in the catalog; the quoted DROP then silently finds
    # nothing, leaving an orphaned index that cannot be dropped programmatically.
    q_index_name = quote_ident(index_name)

    op_class = get_vector_opclass(metric)

    if index_type == "ivfflat":
        lists = index_params.get("lists", 100)
        # IVFFlat requires proper parentheses for functional indexes with operator class
        # The column_name should already include the dimension cast
        return f"""
        CREATE INDEX IF NOT EXISTS {q_index_name}
        ON {q_schema_name}.{q_table_name}
        USING ivfflat (({column_name}) {op_class})
        WITH (lists = {lists})
        """
    elif index_type == "hnsw":
        m = index_params.get("m", 16)
        ef_construction = index_params.get("ef_construction", 200)
        # HNSW index (requires pgvector 0.5.0+)
        # The column_name should already include the dimension cast
        return f"""
        CREATE INDEX IF NOT EXISTS {q_index_name}
        ON {q_schema_name}.{q_table_name}
        USING hnsw (({column_name}) {op_class})
        WITH (m = {m}, ef_construction = {ef_construction})
        """
    else:
        # Default to basic index
        return f"""
        CREATE INDEX IF NOT EXISTS {q_index_name}
        ON {q_schema_name}.{q_table_name}
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


def get_vector_count_sql(q_schema_name: str, q_table_name: str, field_name: str) -> str:
    """Get SQL to count vectors in a field.

    Args:
        q_schema_name: Pre-quoted schema name (e.g. ``'"public"'``)
        q_table_name: Pre-quoted table name (e.g. ``'"MyTable"'``)
        field_name: Vector field name

    Returns:
        SQL query string
    """
    return f"""
    SELECT COUNT(*) as count
    FROM {q_schema_name}.{q_table_name}
    WHERE data ? '{field_name}'
    """


def get_index_check_sql(
    schema_name: str, table_name: str, field_name: str
) -> tuple[str, list[Any]]:
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
    if hasattr(vector, "tolist"):
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
