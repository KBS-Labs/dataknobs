"""Tests for PostgreSQL vector integration.

These tests use the in-memory backend to test vector functionality
without requiring a real PostgreSQL instance with pgvector.
"""

import pytest

from dataknobs_data import Record, VectorField
from dataknobs_data.backends.memory import AsyncMemoryDatabase

# Skip tests if numpy is not available
np = pytest.importorskip("numpy")


@pytest.mark.asyncio
class TestVectorFieldIntegration:
    """Test vector field integration with real database backends."""

    @pytest.fixture
    async def db(self):
        """Create an in-memory database for testing."""
        db = AsyncMemoryDatabase()
        await db.connect()

        yield db

        await db.close()

    async def test_create_record_with_vector(self, db):
        """Test creating a record with vector field."""
        # Create record with vector
        vector = np.array([0.1, 0.2, 0.3])
        record = Record(data={"text": "sample text"})
        record.fields["embedding"] = VectorField(
            vector, name="embedding", source_field="text", model_name="test-model"
        )

        record_id = await db.create(record)
        assert record_id is not None

        # Read it back
        retrieved = await db.read(record_id)
        assert retrieved is not None
        assert "text" in retrieved.fields
        assert "embedding" in retrieved.fields

        # The vector should be stored as a list in JSON
        embedding_value = retrieved.fields["embedding"].value
        if isinstance(embedding_value, list):
            assert np.allclose(embedding_value, vector.tolist())

    async def test_vector_field_serialization(self, db):
        """Test vector field serialization and deserialization."""
        vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        record = Record(data={"title": "Test Document"})
        record.fields["features"] = VectorField(
            vector,
            name="features",
            dimensions=5,
            source_field="title",
            model_name="bert",
            model_version="1.0",
        )

        # Store and retrieve
        record_id = await db.create(record)
        retrieved = await db.read(record_id)

        assert retrieved is not None
        assert "features" in retrieved.fields

        # Check if vector data is preserved
        features_value = retrieved.fields["features"].value
        if isinstance(features_value, list):
            assert len(features_value) == 5
            assert np.allclose(features_value, vector.tolist())

    async def test_search_with_vector_fields(self, db):
        """Test searching records that contain vector fields."""
        # Create multiple records with vectors
        vectors = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6]), np.array([0.7, 0.8, 0.9])]

        record_ids = []
        for i, vec in enumerate(vectors):
            record = Record(data={"text": f"document {i}", "category": "test"})
            record.fields["embedding"] = VectorField(vec, name="embedding")
            record_id = await db.create(record)
            record_ids.append(record_id)

        # Search by regular field
        from dataknobs_data import Query, Filter, Operator

        query = Query(filters=[Filter(field="category", operator=Operator.EQ, value="test")])

        results = await db.search(query)
        assert len(results) == 3

        # Verify all records have embedding fields
        for record in results:
            assert "embedding" in record.fields

    async def test_update_record_with_vector(self, db):
        """Test updating a record that contains a vector field."""
        # Create initial record
        initial_vector = np.array([1.0, 0.0, 0.0])
        record = Record(data={"text": "initial"})
        record.fields["embedding"] = VectorField(initial_vector, name="embedding")

        record_id = await db.create(record)

        # Update the record with new vector
        updated_vector = np.array([0.0, 1.0, 0.0])
        updated_record = Record(data={"text": "updated"})
        updated_record.fields["embedding"] = VectorField(updated_vector, name="embedding")

        success = await db.update(record_id, updated_record)
        assert success is True

        # Verify update
        retrieved = await db.read(record_id)
        assert retrieved.fields["text"].value == "updated"
        embedding_value = retrieved.fields["embedding"].value
        if isinstance(embedding_value, list):
            assert np.allclose(embedding_value, updated_vector.tolist())

    async def test_batch_operations_with_vectors(self, db):
        """Test batch operations with vector fields."""
        rng = np.random.default_rng(0)
        records = []
        for i in range(5):
            record = Record(data={"index": i})
            record.fields["vector"] = VectorField(
                rng.random(3),  # Random 3D vector
                name="vector",
            )
            records.append(record)

        # Batch create
        ids = await db.create_batch(records)
        assert len(ids) == 5

        # Verify all were created
        for record_id in ids:
            retrieved = await db.read(record_id)
            assert retrieved is not None
            assert "vector" in retrieved.fields


class TestVectorOperations:
    """Test vector-specific operations."""

    def test_vector_field_cosine_similarity(self):
        """Test cosine similarity computation between vector fields."""
        vec1 = VectorField(np.array([1.0, 0.0, 0.0]), name="v1")
        vec2 = VectorField(np.array([1.0, 0.0, 0.0]), name="v2")
        vec3 = VectorField(np.array([0.0, 1.0, 0.0]), name="v3")

        # Same vectors should have similarity 1
        similarity = vec1.cosine_similarity(vec2)
        assert np.isclose(similarity, 1.0)

        # Orthogonal vectors should have similarity 0
        similarity = vec1.cosine_similarity(vec3)
        assert np.isclose(similarity, 0.0)

    def test_vector_field_euclidean_distance(self):
        """Test Euclidean distance computation."""
        vec1 = VectorField(np.array([0.0, 0.0]), name="v1")
        vec2 = VectorField(np.array([3.0, 4.0]), name="v2")

        # Distance should be 5 (3-4-5 triangle)
        distance = vec1.euclidean_distance(vec2)
        assert np.isclose(distance, 5.0)

    def test_vector_field_with_metadata(self):
        """Test vector field with full metadata."""
        vector = np.array([0.1, 0.2, 0.3])
        field = VectorField(
            value=vector,
            name="embedding",
            dimensions=3,
            source_field="text",
            model_name="bert-base",
            model_version="1.0.0",
            metadata={"custom_key": "custom_value"},
        )

        assert field.dimensions == 3
        assert field.source_field == "text"
        assert field.model_name == "bert-base"
        assert field.model_version == "1.0.0"
        assert field.metadata["custom_key"] == "custom_value"

        # Convert to dict and back
        data = field.to_dict()
        restored = VectorField.from_dict(data)

        assert restored.dimensions == field.dimensions
        assert restored.source_field == field.source_field
        assert restored.model_name == field.model_name
        assert np.allclose(restored.value, field.value)


class TestPostgresVectorUtilities:
    """Test PostgreSQL vector utility functions."""

    def test_format_vector_for_postgres(self):
        """Test formatting vectors for PostgreSQL."""
        from dataknobs_data.backends.postgres_vector import format_vector_for_postgres

        # Test numpy array
        vector = np.array([0.1, 0.2, 0.3])
        result = format_vector_for_postgres(vector)
        assert result == "[0.1,0.2,0.3]"

        # Test list
        vector = [1.0, 2.0, 3.0]
        result = format_vector_for_postgres(vector)
        assert result == "[1.0,2.0,3.0]"

    def test_parse_postgres_vector(self):
        """Test parsing PostgreSQL vector strings."""
        from dataknobs_data.backends.postgres_vector import parse_postgres_vector

        result = parse_postgres_vector("[0.1,0.2,0.3]")
        assert result == [0.1, 0.2, 0.3]

        result = parse_postgres_vector("[]")
        assert result == []

    def test_get_vector_operator(self):
        """Test getting correct PostgreSQL operators.

        The last assertion used to be ``get_vector_operator("unknown") ==
        "<=>"  # Default``, which pinned the defect: the fallback answered
        every unrecognised name --- including ``dot_product`` and ``l1``,
        two real ``DistanceMetric`` members the table had missed --- with
        cosine distances. Exhaustiveness over the enum is in
        ``test_distance_metric_vocabulary.py``; this cell keeps the spot
        check and the refusal.
        """
        from dataknobs_data.backends.postgres_vector import get_vector_operator

        assert get_vector_operator("cosine") == "<=>"
        assert get_vector_operator("euclidean") == "<->"
        assert get_vector_operator("inner_product") == "<#>"
        assert get_vector_operator("dot_product") == "<#>"
        assert get_vector_operator("l2") == "<->"
        assert get_vector_operator("l1") == "<+>"
        with pytest.raises(ValueError, match="unknown"):
            get_vector_operator("unknown")

    def test_get_optimal_index_type(self):
        """Test optimal index selection based on dataset size."""
        from dataknobs_data.backends.postgres_vector import get_optimal_index_type

        # Small dataset
        index_type, params = get_optimal_index_type(1000)
        assert index_type == "ivfflat"
        assert params["lists"] == 100

        # Medium dataset
        index_type, params = get_optimal_index_type(100000)
        assert index_type == "ivfflat"
        assert params["lists"] > 100

        # Large dataset
        index_type, params = get_optimal_index_type(10000000)
        assert index_type == "hnsw"
        assert "m" in params
        assert "ef_construction" in params

    def test_build_vector_index_sql(self):
        """Test building vector index SQL."""
        from dataknobs_data.backends.postgres_vector import build_vector_index_sql

        # IVFFlat index
        sql = build_vector_index_sql(
            "records",
            "public",
            "vector_embedding",
            768,
            metric="cosine",
            index_type="ivfflat",
            index_params={"lists": 100},
        )
        assert "CREATE INDEX" in sql
        assert "USING ivfflat" in sql
        assert "vector_cosine_ops" in sql  # Cosine operator class
        assert "lists = 100" in sql

        # HNSW index
        sql = build_vector_index_sql(
            "records",
            "public",
            "vector_embedding",
            768,
            metric="euclidean",
            index_type="hnsw",
            index_params={"m": 16, "ef_construction": 200},
        )
        assert "USING hnsw" in sql
        assert "vector_l2_ops" in sql  # L2 operator class
        assert "m = 16" in sql
        assert "ef_construction = 200" in sql


class TestTheFieldNameNeverReachesSqlUnvalidated:
    """A JSONB key lands in a SQL *string literal*, where quoting does not apply.

    ``build_vector_value_expression``'s docstring states the rule --- the
    serializer validates the field name because ``quote_ident`` cannot reach
    a string-literal position --- and ``get_vector_count_sql``, in the same
    module, interpolated it raw:

        WHERE data ? '{field_name}'

    It had one caller on the async twin; this pass gave it two more on the
    sync twin, and on both of those the call runs before anything validates.
    ``get_vector_index_stats`` validates nowhere at all and wraps its whole
    body in ``except Exception``, so a statement that ran and a statement
    that failed are reported the same way.
    """

    INJECTION = "e'; DROP TABLE users; --"

    def test_the_count_query_refuses_an_unsafe_field_name(self):
        from dataknobs_data.backends.postgres_vector import get_vector_count_sql

        with pytest.raises(ValueError, match="Invalid field name"):
            get_vector_count_sql('"public"', '"records"', self.INJECTION)

    def test_the_index_check_refuses_an_unsafe_field_name(self):
        """Its siblings bind, so this one is symmetry rather than exposure.

        Stated anyway: the two are called together, from the same method, on
        the same argument, and a reader who finds one validated and one not
        has to work out which position each occupies before trusting either.
        """
        from dataknobs_data.backends.postgres_vector import get_index_check_sql

        with pytest.raises(ValueError, match="Invalid field name"):
            get_index_check_sql("public", "records", self.INJECTION)

    def test_a_legitimate_field_name_still_works(self):
        from dataknobs_data.backends.postgres_vector import get_vector_count_sql

        sql = get_vector_count_sql('"public"', '"records"', "embedding")
        assert "data ? 'embedding'" in sql

    @pytest.mark.parametrize(
        "builder,args",
        [
            ("get_vector_count_sql", ('"public"', '"records"')),
            ("get_index_check_sql", ("public", "records")),
        ],
    )
    def test_no_builder_in_the_module_takes_a_field_name_on_trust(self, builder, args):
        """The property, rather than the two sites that currently hold it."""
        import dataknobs_data.backends.postgres_vector as module

        with pytest.raises(ValueError):
            getattr(module, builder)(*args, "embedding; DROP TABLE users")


class TestTheSearchPredicateExcludesWhatCannotBeAVector:
    """``data ? field`` is a *key presence* test, and a key can hold null.

    The async twin's old SQL said ``WHERE vector_<field> IS NOT NULL``, which
    is structurally immune: a column either holds a vector or does not. The
    shared builder that replaced it tests that the JSON object has the key,
    and ``{"embedding": null}`` has the key. ``record_to_json`` writes exactly
    that for a record carrying a ``None`` value.

    Such a row yields ``distance = NULL``, Postgres sorts NULLs last under
    ``ORDER BY distance``, and so it surfaces only when the corpus holds fewer
    than ``k`` real vectors --- at which point ``float(row["distance"])``
    raises ``TypeError`` and the whole search fails rather than the one row.
    A non-vector *string* under the key is worse: the cast raises inside
    Postgres and no row comes back at all.
    """

    def _sql(self, **overrides):
        from dataknobs_data.backends.postgres_vector import build_vector_search_sql

        kwargs = {
            "q_qualified": '"public"."records"',
            "vector_field": "embedding",
            "dimensions": 3,
            "metric": "cosine",
            "vector_placeholder": "$1",
            "field_placeholder": "$2",
            "limit_clause": "LIMIT 10",
        }
        kwargs.update(overrides)
        return build_vector_search_sql(**kwargs)

    def test_the_predicate_requires_a_json_array_or_object(self):
        sql = self._sql()
        assert "jsonb_typeof(data->'embedding') IN ('array', 'object')" in sql, (
            "key presence alone admits {'embedding': null} and "
            "{'embedding': 'not a vector'}, neither of which casts to a vector"
        )

    def test_the_key_presence_test_is_kept_beside_it(self):
        """It binds the field name and is what a GIN index on ``data`` can serve."""
        assert "data ? $2" in self._sql()


class TestTheHybridArmsReturnTheRowsTheyRanked:
    """``LIMIT`` without ``ORDER BY`` takes an arbitrary ``fetch_k``.

    Each arm computes a ``ROW_NUMBER()`` over an ``ORDER BY`` in its window
    clause, which ranks the whole set --- and then a bare ``LIMIT fetch_k``
    keeps whichever rows the executor happened to produce first. The ranks are
    correct; the rows they are attached to are a sample. Postgres is free to
    return the thirty rows it read first, so the fusion can rank a corpus's
    worst matches against each other and report them as the best.

    It was masked while the vector arm's predicate was ``vector_<field> IS
    NOT NULL``, which matched only what three of fourteen write paths filled
    --- usually fewer rows than ``fetch_k``, so the ``LIMIT`` never bound.
    Pointing both arms at the ``data`` column every write path fills is what
    makes it reachable: the predicate now matches the whole corpus.
    """

    def _sql(self, **overrides):
        from dataknobs_data.backends.postgres_vector import build_hybrid_search_sql

        kwargs = {
            "q_qualified": '"public"."records"',
            "text_concat": "COALESCE(data->>'content', '')",
            "vector_field": "embedding",
            "dimensions": 3,
            "metric": "cosine",
            "fetch_k": 30,
            "text_placeholder": "$1",
            "vector_placeholder": "$2",
            "field_placeholder": "$3",
        }
        kwargs.update(overrides)
        return build_hybrid_search_sql(**kwargs)

    @staticmethod
    def _clauses(sql: str) -> list[str]:
        """The tokens of each arm, in order, so ordering can be read off."""
        import re

        return re.findall(r"\bORDER BY\b|\bLIMIT\b|\bAS (?:text_rank|vector_rank)\b", sql)

    def test_every_limit_is_preceded_by_an_order_by_of_its_own(self):
        """Not the window's ``ORDER BY`` --- the statement's.

        Read in order, each arm must show its window ordering, the rank it
        names, and then an ordering of the arm itself before its ``LIMIT``.
        A ``LIMIT`` reached with no ``ORDER BY`` since the last rank is the
        defect.
        """
        tokens = self._clauses(self._sql())
        seen_order_since_rank = False
        for token in tokens:
            if token.startswith("AS "):
                seen_order_since_rank = False
            elif token == "ORDER BY":
                seen_order_since_rank = True
            elif token == "LIMIT":
                assert seen_order_since_rank, (
                    "a hybrid arm truncates with LIMIT before ordering its own "
                    f"rows; the arms read: {tokens}"
                )

    def test_the_arms_order_by_the_quantity_they_rank_on(self):
        sql = self._sql()
        assert "ORDER BY text_score DESC" in sql
        assert "ORDER BY vector_distance" in sql

    def test_a_filter_reaches_both_arms(self):
        """``filter`` was accepted by ``hybrid_search`` and bound into nothing.

        The native path built its parameter list as ``[text, vector, field]``
        and never consulted the argument, so a filtered hybrid search searched
        the whole table and said nothing. Both arms have to carry it or the
        fusion joins a filtered ranking to an unfiltered one.
        """
        sql = self._sql(filter_clause="AND data->>'tenant' = $4")
        assert sql.count("AND data->>'tenant' = $4") == 2

    def test_the_vector_arm_excludes_what_cannot_be_a_vector(self):
        """Same predicate as the k-NN search, for the same reason."""
        assert "jsonb_typeof(data->'embedding') IN ('array', 'object')" in self._sql()


class TestTheIndexCheckMatchesThisClassesIndexes:
    """``LIKE '%<field>%'`` matches the index this change orphaned.

    ``_ensure_vector_column`` used to add a ``vector_<field>`` column and
    build an index over it, named from the *column* --- so on a table written
    by an earlier release the catalog holds
    ``idx_<table>_"vector_embedding"_cosine``. Nothing reads that column any
    more, but ``%embedding%`` matches its name, so
    ``get_vector_index_stats`` reports ``indexed: True`` for a field whose
    searches are running a sequential scan. A stats call exists to answer
    exactly that question, and on every upgraded table it answered it wrong.

    A prefix pattern would not have been enough either: ``_`` is a single-
    character wildcard in ``LIKE`` and these names are full of them, so
    ``idx_records_embedding_%`` still claims ``embedding_v2``'s index. The
    check names the indexes this class builds, exactly.
    """

    def _names(self, field="embedding", table="records"):
        from dataknobs_data.backends.postgres_vector import get_index_check_sql

        _, params = get_index_check_sql("public", table, field)
        return set(params[2])

    def test_every_index_this_class_can_build_is_named(self):
        from dataknobs_data.backends.postgres_vector import get_vector_index_name
        from dataknobs_data.vector.types import DistanceMetric

        names = self._names()
        for member in DistanceMetric:
            built = get_vector_index_name("records", "embedding", member.canonical().value)
            assert built in names, (
                f"{member} builds {built!r} and the index check does not look for it"
            )

    def test_one_name_per_family_rather_than_per_spelling(self):
        """Six members, four indexes --- the aliases cannot name a fifth."""
        assert len(self._names()) == 4

    def test_the_orphaned_column_index_is_not_named(self):
        from dataknobs_data.backends.postgres_vector import (
            extract_field_name,
            get_vector_index_name,
        )
        from dataknobs_utils.sql_utils import quote_ident

        legacy = get_vector_index_name(
            "records", extract_field_name(quote_ident("vector_embedding")), "cosine"
        )
        assert legacy not in self._names(), (
            f"{legacy!r} indexes a column nothing reads and is reported as this field's index"
        )

    def test_another_fields_index_is_not_named(self):
        from dataknobs_data.backends.postgres_vector import get_vector_index_name

        assert get_vector_index_name("records", "embedding_v2", "cosine") not in self._names()
